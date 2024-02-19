# 💫 HTTP Rule

## ✔️ Method

- GET
    - 리소스 본문(body)까지 조회
- HEAD
    - 리소스 헤더(header) 정보만 받아옴
    - 메타 데이터 이용 시 사용할 수 있는 메소드 (→ 캐싱과 연관)
- POST
    - 주로 ‘등록’에 사용
    - 프로세스 처리
        - 단순한 값 변경을 넘어
        - 주문에서 결제완료 → 배달시작 → 배달완료와 같이
        - 프로세스의 상태가 변하는 경우
    - 애매하면 POST
- PUT
    - 리소스 **전체** 대체
    - 없는 경우 새로 생성
- PATCH
    - 리소스 **부분** 변경
- DELETE
    - 리소스 **전체** 삭제

> **리소스란?**

- ”미네랄을 캐라”에서 ‘미네랄’에 해당
- 회원 등록/조회/수정/삭제에서 ‘회원’에 해당
- 리소스가 단일 실제 데이터 항목을 기반으로 할 필요는 없음
- 예를 들어 주문 리소스는 내부적으로 관계형 데이터베이스의 여러 테이블로 구현할 수 있지만, 클라이언트에 대해서는 단일 엔터티로 표시됨
- 단순히 데이터베이스의 내부 구조를 반영하는 API를 만들지 말기
- REST의 목적은 엔터티 및 해당 엔터티에서 애플리케이션이 수행할 수 있는 작업을 모델링하는 것❗️
- 클라이언트는 내부 구현에 노출되면 안 됨
- 리소스 URI를 컬렉션/항목/컬렉션보다 더 복잡하게 요구하지 않는 것이 좋음

> **POST vs PUT?**

- PUT은 클라이언트가 리소스 위치를 알고 URI 지정
- HTML Form 형태는 GET과 POST만 지원하기 때문에 PUT 부분도 URI에 동작 설명 추가하여 POST로 사용
- POST는 멱등이 아님

> **캐시?**

- GET, HEAD, POST, PATCH 캐시가능
- 실제로는 GET, HEAD 정도만 캐시로 사용
- POST, PATCH는 본문 내용까지 캐시 키로 고려해야 하는데, 구현이 쉽지 않음

## ✔️ URI

- URI는 **리소스**만 식별
- 행위는 **메소드**가 구분
- 예시
    - **회원** 목록 조회 → /members → GET
    - **회원** 조회 → /members/{id} → GET
    - **회원** 등록 → /members/{id} → POST
    - **회원** 수정 → /members/{id} → PATCH, PUT, POST
    - **회원** 삭제 → /members/{id} → DELETE
    - 필요한 경우 URI 추가 사용하여 구분

> **복수단어?**
- 계층 구조상 상위를 컬렉션으로 보고 복수단어 사용 권장(member -> members)

> **컬렉션?**
- 서버가 관리하는 리소스 디렉토리

## ✔️ 상태 코드

- 200(OK)
    - 요청 성공
- 204(콘텐츠 없음)
    - 요청이 처리되었지만 HTTP 응답에 포함된 응답 본문이 없는 경우
- 201(만들어짐)
    - PUT 메서드와 POST 메서드, 새 리소스를 만드는 경우
- 202(수락됨)
    - 요청 처리가 수락되었지만 아직 완료되지 않았음을 나타내는 경우
    - POST, PUT, PATCH 또는 DELETE 작업을 완료하는 데 시간이 걸리는 처리가 필요한 경우
    - 처리 작업이 완료될 때까지 기다렸다가 클라이언트에 응답을 보내는 경우 허용되지 않는 수준의 대기 시간이 발생할 경우
        - 비동기 작업을 수행하는 방안을 고려해 보아야

# 💫 응답 Rule

## ✔️ 정상 처리

- dto가 없는 경우
    - 상태 코드(204)만 반환

    ```java
    return ResponseEntity.noContent().build();
    ```

- dto가 있는 경우
    - SuccessResponse로 감싸서 반환
        - API 응답 형식 통일 가능 : 코드 통일성, 프론트 처리 용이
        - 공통 데이터 처리 용이
    - SuccessResponse 필드
        - status : boolean, true 고정
        - data : T, 개별 dto 정의

    ```java
    @Getter
    @Schema(description = "성공 Response")
    @NoArgsConstructor(access = AccessLevel.PRIVATE)
    public class SuccessResponse<T> {
    
        @Schema(description = "성공 여부. 항상 true 이다.", defaultValue = "true")
        private final boolean status = true;
        private T data;
    
        public static <T> SuccessResponse<T> of(T data) {
            SuccessResponse<T> SuccessResponse = new SuccessResponse<>();
            SuccessResponse.data = data;
            return SuccessResponse;
        }
    
        public ResponseEntity<SuccessResponse<T>> asHttp(HttpStatus httpStatus) {
            return ResponseEntity.status(httpStatus).body(this);
        }
    }
    ```

    ```java
    return SuccessResponse.of(memberService.getMyPage(sessionId))        
    .asHttp(HttpStatus.OK);
    ```


## ✔️ 예외 처리

- 커스텀 예외
    - BusinessException.class 상속

    ```java
    @Getter
    public class BusinessException extends RuntimeException {
    
        private final ErrorCode errorCode;
    
        public BusinessException(ErrorCode errorCode) {
            super(errorCode.getMessage());
            this.errorCode = errorCode;
        }
    }
    ```

    - ErrorCode(http status, code, message) 개별 정의

    ```java
    @Getter
    @RequiredArgsConstructor
    public enum ErrorCode {
    
        // Common
        INTERNAL_SERVER_ERROR(500, "C001", "서버에 오류가 발생하였습니다."),
    
        // Member
        MEMBER_NOT_FOUND(404,"M001","사용자를 찾을 수 없습니다."),
        ACCESS_FORBIDDEN(403, "M002", "접근 권한이 없는 계정입니다."),
        S3_INVALID(500, "M003", "이미지 업로드에 실패하였습니다.");
    
        private final int status;
        private final String code;
        private final String message;
    }
    ```

- 기존 예외
    - Exception.class
    - 500(internal server error) 반환
- GlobalExceptionHandler
    - 발생한 예외 처리

    ```java
    @RestControllerAdvice
    @Slf4j
    public class GlobalExceptionHandler extends ResponseEntityExceptionHandler {
    
        @ExceptionHandler(BusinessException.class)
        public ResponseEntity<?> handleBusinessException(BusinessException e) {
            ErrorCode errorCode = e.getErrorCode();
            if (errorCode.getStatus() == HttpStatus.INTERNAL_SERVER_ERROR.value()) {
                log.error("handleBusinessException", e);
            } else {
                log.warn("handleBusinessException", e);
            }
            return makeErrorResponse(errorCode);
        }
    
        @ExceptionHandler(Exception.class)
        public ResponseEntity<?> handleException(Exception e) {
            log.error("handleException", e);
            return makeErrorResponse(ErrorCode.INTERNAL_SERVER_ERROR);
        }
    
        private ResponseEntity<?> makeErrorResponse(ErrorCode errorCode) {
            return ResponseEntity.status(errorCode.getStatus())
                    .body(ErrorResponse.of(errorCode));
        }
    }
    ```

    - ErrorResponse 반환
    - ErrorResponse 필드
        - status : boolean, false 고정
        - code: String, ErrorCode 통해 개별 정의

    ```java
    @Getter
    @Schema(description = "실패 Response")
    @RequiredArgsConstructor(access = AccessLevel.PRIVATE)
    public class ErrorResponse {
    
        @Schema(description = "성공 여부. 항상 false 이다.", defaultValue = "false")
        private final boolean status = false;
        private final String code;
    
        public static ErrorResponse of(ErrorCode errorCode) {
            return new ErrorResponse(errorCode.getCode());
        }
    }
    ```


# 💫 DTO Rule

## ✔️ Request

```java
@Getter
@Schema(description = "한 줄 소개 변경 Request")
@RequiredArgsConstructor
@NoArgsConstructor(force = true)
public class BioRequest {
    @NotNull
    private final Long id;
    private final String bio;
}
```

- `final` : DTO는 단순히 값 저장 → 전달 역할이기 때문에 값의 변경이 필요 없다!
- `@NoArgsConstructor`: 리플렉션에 사용
    - 리플렉션?
        - 구체적인 클래스 타입을 알지 못해도 클래스의 메소드, 필드 등에 접근 가능한 Java API
        - 접근 제어자 상관 X
        - 런타임 시점에 실행되는 클래스의 동적 객체 생성 시 필요
        - 이때 생성자의 인자는 알 수 없음 → 기본 생성자 이용 → 기본 생성자 필요
    - `JSON → 객체` 역직렬화시 리플렉션 이용
    - 그렇지 않은 경우 `@JsonProperty`를 사용해야 함
        - 필드가 많아진다면?
        - JSON이 아니라 다른 포맷을 사용한다면?
- `@NoArgsConstructor(force=true)`: 필드 강제 기본 값 초기화
    - final 변수가 있을 경우 초기화 없이 기본 생성자 생성 시 컴파일 에러 발생
- `@RequiredArgsConstructor`
    - 클래스에 **`final`** 키워드가 붙은 필드들을 가지는 생성자를 자동으로 생성
    - 실제 필드 값 적용하여 객체 생성
    - 불변성 유지

## ✔️ Response

```java
@Getter
@Schema(description = "설정 페이지 Response")
@Builder(access = AccessLevel.PRIVATE)
@RequiredArgsConstructor(access = AccessLevel.PRIVATE)
public class SettingPageResponse {
    private final String name;
    @Schema(description = "S3 저장소 내 프로필 사진 고유 코드", defaultValue = "")
    private final String urlCode;

    public static SettingPageResponse of(Member member) {
        return SettingPageResponse.builder()
                .name(member.getName())
                .urlCode(member.getUrlCode())
                .build();
    }
}
```

- 필드에 사용된 @Schema는 부가 설명이 필요할 시 임의로 설정
- @Builder, @RequiredArgsConstructor는 접근 레벨을 private으로 설정하여 클래스 외부에서 객체가 생성되는 것을 막음
    - 객체 생성의 남발을 막고
    - 전달할 값을 저장한다는 DTO의 역할과 책임을 클래스에 부여, 객체 지향적(다만 관점의 차이 있음)

&nbsp;
&nbsp;
&nbsp;
> 참고 : <모든 개발자를 위한 HTTP 웹 기본 지식 강의 - 인프런> ,  
> https://github.com/InsuranceSystem/InsuranceSystem2,  
> https://learn.microsoft.com/ko-kr/azure/architecture/best-practices/api-design
