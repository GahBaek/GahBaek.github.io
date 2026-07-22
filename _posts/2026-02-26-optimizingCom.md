---
layout: posts
title: "Optimizing Compiler"
categories: ["Compiler"]
---

### Optimizing Compiler

![image.png](attachment:4548da00-aa5b-48f6-a1e7-3d8c74635bc2:image.png)

**Middle End**

: IR 을 분석해서 IR 자체를 더 좋은 IR 로 바꾸는 최적화 단계이다.

즉, Back End 가 기계어로 바꾸기 좋은 형태가 되도록 IR 을 정리하고 개선한다.

**최적화의 목표**

- execution time 감소
- space usage (메모리 / 코드 크기) 감소
- power consumption 감소
- 프로그램의 의미 (semantics)는 보존해야한다.

### Scanner in Front End

![image.png](attachment:dcef8418-7d9e-4ea3-aff8-9ac1cf98a40f:image.png)

**Scanner**

Scanner 는 Front End의 첫 단계이다. 

token 은 Parser 가 처리하기 좋은 단위이다.

⇒ 소스 코드를 문자 단위로 읽어서 토큰으로 쪼갠다.

**Scanner Generator**

![image.png](attachment:9128db29-b076-4931-bca2-d74d90ad9bdd:image.png)

: “Scanner 를 손으로 짜기 싫다 → 자동으로 만들자”

왜 자동 생성이 가능한가 ?

Scanner 가 하는 pattern matching 은 대부분 정규 표현식으로 표현 가능하다.

→ 도구에 규칙을 주면 도구가 Scanner 코드 (또는 table) 을 만들어 준다.

**Build tables and code from a DFA**

정규 표현식 → NFA → DFA 로 변환한 뒤, DFA 를 빠르게 실행할 수 있게 전이 테이블 형태로 만들거나, 그걸 기반으로 C/JAva 코드를 생성한다.

- DFA 는 입력을 한 글자씩 읽으면서 현재 상태를 업데이트하는 구조라 Scanner 가 보통 O(n) 시간에 매우 빠르게 돈다.

**Flow**

실행 시점)

source code → Scanner → tokens

생성 시점 (컴파일러 만들 때)

specifications → Scanner Generator → tables or code → 이게 Scanner 가 된다.

개발자는 token 정의 (정규식 + 액션)만 작성

Generator 가 Scanner 구현을 만들어 준다.

Scanner 가 없다면

> 한 글자씩 읽기, 공백/주석 skip, 키워드 vs 식별자 구분, 숫자/문자 literal 처리 등등을 구현해야한다.
> 

따라서 규칙만 적으면 자동으로 코드를 만들어주는 generator 를 사용하는 것이다.

**Scanner 와 Scanner Generator 의 차이**

**Scanner**

입력: 소스 코드 문자들

출력: token 들

즉, 실제로 compile 할 때 실행되는 프로그램이 Scanner 이다.

**Scanner Generator**

입력: token 들을 어떻게 인식할지 적어놓은 규칙

출력: 그 규칙대로 동작하는 Scanner 코드 또는 table

즉, scanner 를 만드는 도구

**정규 표현식 → NFA → DFA 를 스캐너가 빠르게 돌기 위해 정규식을 실행 가능한 기계로 바꾸는 과정**

왜 굳이 변환 ?

정규 표현식은 패턴 설명이라서 그대로는 애매하다. 컴퓨터가 소스코드를 한 글자씩 읽을 때마다

- 지금까지 읽은 게 이 정규식에 맞나 ?
- 앞으로 더 읽으면 맞을 수도 있나 ?
- 여기서 토큰을 끝내야 하나 ?

이걸 빠르게 판단해야한다.

그래서 정규식을 상태들을 가진 기계로 바꿔서, 입력을 한 글자씩 넣으며 상태를 이동시키는 방식으로 처리한다.

1. **정규 표현식 → NFA (Non-deterministic Finite Automation)**

NFA: 갈림길이 있는 state 기계

1. **NFA → DFA (Deterministic Finite Automation)**

DFA: 갈림길이 없는 state 기계 (lexer 가 선호)

⇒ 실제로 스캐너는 source code 를 엄청 길게 읽어야 하므로 매 글자마다 갈림길들을 동시에 추적하면 비싸다. DFA 는 갈림길이 없어서 매 글자를 O(1) 로 빠르게 처리가 가능하다.

1. **NFA 에서 DFA 로 갈림길을 없애는 핵심 아이디어**

NFA 의 여러 상태를 동시에 있을 수 있다는 상황을 DFA 에서는 그 여러 상태의 집합을 하나의 상태로 합쳐서 표현한다.
