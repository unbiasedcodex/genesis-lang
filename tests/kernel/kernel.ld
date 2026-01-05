/* Genesis Kernel Linker Script - x86_64 */
/* For integration testing of pre-kernel features */

OUTPUT_FORMAT(elf64-x86-64)
OUTPUT_ARCH(i386:x86-64)
ENTRY(_start)

SECTIONS {
    /* Start at 1MB - standard location for kernel */
    . = 1M;

    /* Code section */
    .text BLOCK(4K) : ALIGN(4K) {
        *(.text.boot)    /* Boot code first */
        *(.text .text.*)
    }

    /* Read-only data */
    .rodata BLOCK(4K) : ALIGN(4K) {
        *(.rodata .rodata.*)
    }

    /* Initialized data */
    .data BLOCK(4K) : ALIGN(4K) {
        *(.data .data.*)
    }

    /* BSS - uninitialized data */
    .bss BLOCK(4K) : ALIGN(4K) {
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        __bss_end = .;
    }

    /* End of kernel */
    __kernel_end = .;

    /* Discard unwanted sections */
    /DISCARD/ : {
        *(.comment)
        *(.note*)
        *(.eh_frame*)
    }
}
