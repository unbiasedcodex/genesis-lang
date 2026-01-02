//! Utility functions for LSP operations
//!
//! Provides conversion between Genesis spans (byte offsets) and LSP positions (line/column).

use genesis::span::Span;
use ropey::Rope;
use tower_lsp::lsp_types::{Position, Range};

/// Line information cache for efficient span-to-position conversion
#[derive(Debug, Clone)]
pub struct LineIndex {
    /// Cumulative byte offsets for each line start
    line_starts: Vec<usize>,
    /// The source text as a rope for efficient operations
    rope: Rope,
}

impl LineIndex {
    /// Create a new line index from source text
    pub fn new(text: &str) -> Self {
        let rope = Rope::from_str(text);
        let mut line_starts = Vec::with_capacity(rope.len_lines());

        let mut offset = 0;
        for line in text.lines() {
            line_starts.push(offset);
            offset += line.len() + 1; // +1 for newline
        }

        // Handle case where text doesn't end with newline
        if line_starts.is_empty() || offset != text.len() + 1 {
            if line_starts.is_empty() {
                line_starts.push(0);
            }
        }

        Self { line_starts, rope }
    }

    /// Convert a byte offset to an LSP Position (0-indexed line and column)
    pub fn offset_to_position(&self, offset: usize) -> Position {
        // Find the line containing this offset
        let line = match self.line_starts.binary_search(&offset) {
            Ok(line) => line,
            Err(line) => line.saturating_sub(1),
        };

        let line_start = self.line_starts.get(line).copied().unwrap_or(0);
        let column = offset.saturating_sub(line_start);

        Position {
            line: line as u32,
            character: column as u32,
        }
    }

    /// Convert an LSP Position to a byte offset
    pub fn position_to_offset(&self, position: Position) -> usize {
        let line = position.line as usize;
        let column = position.character as usize;

        if line >= self.line_starts.len() {
            return self.rope.len_bytes();
        }

        let line_start = self.line_starts[line];
        let line_end = self.line_starts
            .get(line + 1)
            .map(|&s| s.saturating_sub(1))
            .unwrap_or_else(|| self.rope.len_bytes());

        (line_start + column).min(line_end)
    }

    /// Convert a Genesis Span to an LSP Range
    pub fn span_to_range(&self, span: Span) -> Range {
        Range {
            start: self.offset_to_position(span.start),
            end: self.offset_to_position(span.end),
        }
    }

    /// Convert an LSP Range to a Genesis Span
    pub fn range_to_span(&self, range: Range) -> Span {
        Span {
            start: self.position_to_offset(range.start),
            end: self.position_to_offset(range.end),
        }
    }

    /// Check if an offset is within a span
    pub fn offset_in_span(&self, offset: usize, span: Span) -> bool {
        offset >= span.start && offset < span.end
    }

    /// Check if a position is within a span
    pub fn position_in_span(&self, position: Position, span: Span) -> bool {
        let offset = self.position_to_offset(position);
        self.offset_in_span(offset, span)
    }

    /// Get the text content of a span
    pub fn span_text(&self, span: Span) -> String {
        let start_char = self.rope.byte_to_char(span.start.min(self.rope.len_bytes()));
        let end_char = self.rope.byte_to_char(span.end.min(self.rope.len_bytes()));
        self.rope.slice(start_char..end_char).to_string()
    }

    /// Get the total length in bytes
    pub fn len(&self) -> usize {
        self.rope.len_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_index() {
        let text = "fn main() {\n    let x = 5\n}";
        let index = LineIndex::new(text);

        // Start of first line
        assert_eq!(index.offset_to_position(0), Position { line: 0, character: 0 });

        // "main" starts at offset 3
        assert_eq!(index.offset_to_position(3), Position { line: 0, character: 3 });

        // Start of second line (after "fn main() {\n")
        assert_eq!(index.offset_to_position(12), Position { line: 1, character: 0 });
    }

    #[test]
    fn test_span_to_range() {
        let text = "fn test() {}";
        let index = LineIndex::new(text);

        let span = Span::new(3, 7); // "test"
        let range = index.span_to_range(span);

        assert_eq!(range.start, Position { line: 0, character: 3 });
        assert_eq!(range.end, Position { line: 0, character: 7 });
    }
}
