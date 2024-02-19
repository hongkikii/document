### @Transactional

- 아래 주소의 블로그에 정리
- https://hongkiki32399.tistory.com/139

### @Valid @Validated

- 서버에서 검증해야 하는 이유
    - 컨트롤러의 중요한 역할중 하나는 HTTP 요청이 정상인지 검증하는 것
    - 클라이언트 검증은 조작할 수 있으므로 보안에 취약
    - 서버만으로 검증하면, 즉각적인 고객 사용성이 부족
    - 둘을 적절히 섞어서 사용하되, 최종적으로 서버 검증은 필수
    - API 방식을 사용하면 API 스펙을 잘 정의해서 검증 오류를 API 응답 결과에 잘 남겨주어야 함
    - 그런데 HTTP 요청은 언제든지 악의적으로 변경해서 요청할 수 있으므로 서버에서 항상 검증해야
    - 예를 들어서 HTTP 요청을 변경해서 item 의 id 값을 삭제하고 요청할 수도 있음
    - 따라서 최종 검증은 서버에서 진행하는 것이 안전
- 이론
    - https://mangkyu.tistory.com/174
- cf. @NonNull vs @NotNull
    - NotNull은 Bean Validation 검증할 때
    - NonNull은 Lombok에서 해당 필드를 초기화해야 한다는 것을 Builder에 요구
