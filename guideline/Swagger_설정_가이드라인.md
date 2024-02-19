## 환경
`Java 17`, `Spring Boot 3.1.1`, `Gradle`

## build.gradle
```
implementation 'org.springdoc:springdoc-openapi-starter-webmvc-ui:2.0.2'
```

## Configuration
```
@OpenAPIDefinition(
        info = @Info(title = "JoA",
        description = "우주 정복 앱 JoA의 API 문서입니다! !",
        version = "v1"),
        servers = @Server(url = "서버_URL", description = "운영 서버")
)
@Configuration
public class SwaggerConfig {

    @Bean
    public GroupedOpenApi allOpenApi() {
        String[] paths = {"/joa/**"};

        return GroupedOpenApi
                .builder()
                .group("전체 API")
                .pathsToMatch(paths)
                .build();
    }

    @Bean
    public GroupedOpenApi authOpenApi() {
        String[] paths = {"/joa/auths/**"};

        return GroupedOpenApi
                .builder()
                .group("사용자 인증 API")
                .pathsToMatch(paths)
                .build();
    }

    @Bean
    public GroupedOpenApi memberProfileOpenApi() {
        String[] paths = {"/joa/member-profiles/**"};

        return GroupedOpenApi
                .builder()
                .group("사용자 정보 API")
                .pathsToMatch(paths)
                .build();
    }

    @Bean
    public GroupedOpenApi heartOpenApi() {
        String[] paths = {"/joa/hearts/**"};

        return GroupedOpenApi
                .builder()
                .group("하트 API")
                .pathsToMatch(paths)
                .build();
    }

    @Bean
    public GroupedOpenApi voteOpenApi() {
        String[] paths = {"/joa/votes/**"};

        return GroupedOpenApi
                .builder()
                .group("투표 API")
                .pathsToMatch(paths)
                .build();
    }

    @Bean
    public GroupedOpenApi locationOpenApi() {
        String[] paths = {"/joa/locations/**"};

        return GroupedOpenApi
                .builder()
                .group("위치 API")
                .pathsToMatch(paths)
                .build();
    }

    @Bean
    public GroupedOpenApi voteReportOpenApi() {
        String[] paths = {"/joa/reports/**"};

        return GroupedOpenApi
                .builder()
                .group("신고 API")
                .pathsToMatch(paths)
                .build();
    }

}
```


## Controller
```
    @Operation(summary = "설정 페이지 정보 조회", description = "설정 페이지에서 필요한 정보 조회 API")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "설정 페이지 필요 정보 반환"),
            @ApiResponse(responseCode = "404", description = "M001: 사용자를 찾을 수 없습니다.", content = @Content(schema = @Schema(hidden = true))),
    })
    @GetMapping("/...")
    public ResponseEntity<...> getSettingPage(
            @Parameter(description = "사용자 세션 id", in = ParameterIn.PATH) @PathVariable("id") Long sessionId) {
        return ...
    }
```

```
    @Operation(summary = "아이디 중복 검증", description = "회원가입 시 중복 아이디가 존재하는지 확인하는 API")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "204", description = "HTTP 상태 코드 반환", content = @Content(schema = @Schema(hidden = true))),
            @ApiResponse(responseCode = "409", description = "M011: 이미 사용 중인 아이디입니다.", content = @Content(schema = @Schema(hidden = true))),

    })
    @PostMapping("/...)
    public ResponseEntity<Void> verify(@RequestBody @Valid LoginIdRequest request) {
        ...
        return ResponseEntity.noContent().build();
    }
```

## DTO
### request
```
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

### response
```
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

## URL
- 서버_URL.swagger-ui.index.html
<br>
<br>
> 참고 : https://github.com/InsuranceSystem/InsuranceSystem2
