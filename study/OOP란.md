# 설계의 악취를 표현하는 용어
- 경직성 : 시스템을 변경하기 어려움, 변경을 하려면 시스템의 다른 부분들까지 많이 변경해야 하기 때문
- 취약성 : 변경을 하면 시스템에서 그 부분과 개념적으로 아무런 관련이 없는 부분이 망가짐
- 부동성 : 시스템을 다른 시스템에서 재사용할 수 있는 컴포넌트로 구분하기 어려움
- 점착성 : 옳은 동작을 하는 것이 잘못된 동작을 하는 것보다 어려움
- 불필요한 복잡성 : 직접적인 효용이 전혀 없는 기반 구조가 설계에 포함되어 있음
- 불필요한 방법 : 단일 추상 개념으로 통합할 수 있는 반복적인 구조가 설계에 포함되어 있음
- 불투명성 : 읽고 이해하기 어려움, 그 의도를 잘 표현하지 못 함

# 단일 책임 원칙(SRP)
  > 한 클래스는 단 한 가지의 변경 이유만을 가져야 한다.
- 한 클래스는 하나의 책임만 져야 한다.
  - **책임**이란? 변경의 이유
- 애플리케이션이 서로 다른 시간에 두 가지 책임의 변경을 유발하는 방식으로 바뀌지 않는다면 분리할 필요는 없다.
- 변경의 축은 실제로 일어날 때만 변경의 축이다.
- 테스트 주도 개발은 책임이 분리되도록 만든다.
- 경직성, 취약성 ⬇️
- 퍼사드, 프록시 패턴

# 개방 폐쇄 원칙(OCP)
  > 소프트웨어 개체(클래스, 모듈, 함수)는 **확장**에 열려 있어야 하고, **수정**에 닫혀 있어야 한다.

- 요구사항 변경에 맞춰 새로운 행위를 추가해 모듈을 변경할 수 있다.
- 모듈의 행위를 확장하는 것이 모듈 코드의 변경을 초래하지는 않는다.
- ✨추상화 : 모듈이 고정된 추상화에 의존한다면, 수정에 닫혀 있을 수 있다❗️
- 자주 변경되는 부분에만 추상화 적용하기
  - 변경이 있을 법한지는 어떻게 알 수 있을까?
    - 적절한 연구, 적절한 질문, 경험과 상식 이용, 그리고 변경이 일어날 때까지 기다리기
    - 불필요한 복잡성을 줄이기 위해
- 어설픈 추상화를 피하는 일은 추상화 자체만큼이나 중요하다.
- 경직성, 부동성 ⬇️
- 템플릿 메소드 패턴, 스트래티지 패턴(구체 클래스 의존성 없애기)

# 리스코프 치환 원칙(LSP)
  > 서브타입(subtype)은 그것의 기반 타입(base type)으로 치환 가능해야 한다.
- 파생 클래스를 만드는 것이 기반 클래스의 변경으로 이어질 때, 대게는 이 설계에 결점이 있음을 의미한다.
  - 이것은 OCP를 위반한 것이다.
- LSP 위반은 잠재적인 OCP 위반이다.
- Square와 Circle이 Shape을 대체할 수 없다는 것은 LSP 위반이며, OCP 위반을 유발한다.
- 수학적으로 직사각형은 정사각형을 포함하는 개념이다.
- 하지만 setWidth(), setHeight() 메소드를 공유할 수 없다!
- IS-A(상속, -이다)는 ‘행위’에 대한 것이다.
- 파생 클래스는 기반 클래스의 행위와 출력 제약을 위반해서는 안 된다.
- 직사각형과 정사각형은 LSP 위반이다.
- 어떤 모델의 유효성은 오직 그 고객의 관점에서만 표현될 수 있다.
- 다만 사용자가 택할 합리적인 가정을 모두 예상할 수는 없다.
  - 오히려 시스템의 불필요한 복잡성만 높인다.
- 취약성의 악취를 맡을 때까지 가장 명백한 LSP 위반을 제외한 나머지의 처리는 연기하는 게 최선이다.
- 그렇다면 어떻게 LSP를 지킬 수 있을까?
- ✨추상 인터페이스 아래 형제 관계로 묶는 계층 구조를 만들자❗️
- LSP는 OCP를 가능하게 하는 주요 요인이다.

# 의존 관계 역전 원칙(DIP)
  > 상위 수준의 모듈은 하위 수준의 모듈에 의존해서는 안 된다. 둘 모두 추상화에 의존해야 한다.

  > 추상화는 구체적인 사항에 의존해서는 안 된다. 구체적인 사항은 추상화에 의존해야 한다.
  
- **역전**이란?
  - 전통적 SW 개발은 상위 모듈이 하위 모듈에, 정책이 구체적인 것에 의존하는 경향이 있다.
  - 잘 설계된 객체 지향 프로그램의 의존성 구조는 전통적인 절차적 방법에 의해 일반적으로 만들어진 의존성 구조가 역전된 것이다.
- 의존성은 이행적(transitive)이다.
- 레이어를 나누자. 단, 하위 수준 레이어는 상위 수준 레이어에 선언된 추상 인터페이스에 의존해야 한다.
- ✨추상화에 의존하자❗️
- 어떤 변수도 구체 클래스에 대한 포인터나 참조값을 가져선 안 된다.
- 어떤 클래스도 구체 클래스에서 파생되어서는 안 된다.
- 어떤 메소드도 그 기반 클래스에서 구현된 메소드를 오버라이드해서는 안 된다.
- 어떤 것이 상위 수준 모듈인가?
  - 시스템 안의 시스템, 메타포

# 인터페이스 분리 원칙(ISP)

  > 클라이언트가 자신이 사용하지 않는 메소드에 의존하도록 강제되어서는 안 된다.

- 불필요한 복잡성, 불필요한 중복성 ⬇️
- 비대한 인터페이스를 가지는 클래스는 응집력이 없는 인터페이스를 가지는 클래스이다.
- 응집력 없는 인터페이스에 필요한 객체가 있다는 것은 인정한다.
- 하지만 클라이언트가 하나의 단일 클래스로 생각하게 해서는 안 된다.
- 클라이언트가 분리되어 있는 경우, 인터페이스도 분리된 상태로 있어야 한다.
- 클라이언트가 자신이 사용하는 인터페이스에 영향을 끼치기 때문이다.
- 경직성, 취약성, 점착성 ⬇️
- 다중 상속을 통해 분리할 수 있다.
- 클라이언트가 호출하는 서비스 메소드에 따라 그룹 지어서 그룹에 따라 분리된 인터페이스를 만들 수 있다.  
&nbsp;
&nbsp;
&nbsp;
> 참고 : <클린 소프트웨어>
