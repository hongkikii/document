# 환경

`AWS Lightsail(ubuntu)`, `Github Actions`, `Gradle`, `Java 17`, `Spring Boot 3.1.1`   

# 1. Lightsail SSH 키 페어 생성하기

- .pem 확장자로 다운로드 가능
- 암호 키 사용 예정

# 2. IAM 사용자 Access-Key 발급받기

- **AWSCodeDeplyFullAccess 역할 적용**
- .csv 파일로 다운로드(한 번만 가능)
- 엑세스 키, 시크릿 키 사용 예정

# 3. Lightsail aws-cli 설치
- configure 설정하기 위해  
```
  sudo apt-get update && sudo apt-get install awscli
  aws --version
```

# 4. Lightsail Configure 설정
```
aws configure
# Access-Key 설정
# Secret-Key 설정
# region 설정
# output format 설정
```

# 5. Github Repository Secrets 등록

- 리포지토리 - Settings - Secrets and Variables
- AWS_ACCESS_KEY_ID : IAM에서 발급받은 엑세스 키
- AWS_SECRET_ACCESS_KEY : IAM에서 발급받은 시크릿 키
- LIGHTSAIL_HOST : Lightsail public IP
- LIGHTSAIL_SSH_KEY : LIghtsail SSH 키페어  
<details>
<summary>SSH 키페어 주의사항</summary>
<div markdown="1">
1.  맥에서는 터미널 `open -e 경로/xxx.pem` 형태로 오픈 가능<br>
2. --BEGIN RSA PRIVATE KEY--—와 --END RSA PRIVATE KEY-—도 포함해야
</div>
</details>


# 6. Github Actions Workflow 생성

- 리포지토리 - Actions - new workflow
- .yml 파일 작성
```
name: CI CD

on:
push:
branches: ['main']
// pull_request:
//   branches: ['main']

env:
LIGHTSAIL_SSH_KEY: ${{ secrets.LIGHTSAIL_SSH_KEY }}
LIGHTSAIL_HOST: ${{ secrets.LIGHTSAIL_HOST }}
LIGHTSAIL_USERNAME: ubuntu
AWS_REGION: ap-northeast-2

jobs:
build:
runs-on: ubuntu-latest

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v3

      - name: Setup Java
        uses: actions/setup-java@v3
        with:
          distribution: 'adopt'
          java-version: '17'

      - name: Build with Gradle
        run: ./gradlew build

      - name: Check Gradle Build Output
        run: ls -la build/libs

      - name: AWS Authentication
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-region: ${{ env.AWS_REGION }}
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Deploy to AWS Lightsail
        uses: appleboy/scp-action@master
        with:
          host: ${{ secrets.LIGHTSAIL_HOST }}
          username: ${{ env.LIGHTSAIL_USERNAME }}
          key: ${{ secrets.LIGHTSAIL_SSH_KEY }}
          source: '소스_파일_경로'
          target: '서비스_경로'

      - name: Restart Application
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.LIGHTSAIL_HOST }}
          username: ${{ env.LIGHTSAIL_USERNAME }}
          key: ${{ secrets.LIGHTSAIL_SSH_KEY }}
          script: |
            chmod +x 서비스_경로/소스_파일_경로
            sudo systemctl restart 서비스_이름.service
```

## 주의 사항
- LIGHTSAIL_USERNAME
    - Lightsail 사이트에서 확인 가능
- 버전
    - 작성일(2024-01-08) 기준
    - Java → @v3
    - AWS → @v2
- application.yml
    - 깃허브 리포지토리에 올라와 있어야
- gradle/wrapper
    - 깃허브 리포지토리에 올라와 있어야
    - gradle-wrapper.jar, gradle-wrapper.properties 전부 필요
- git 서브모듈 설정
  ```
  # 남아있는 캐시 제거
  git ls-files --stage 서브_모듈_경로
  git rm --cached 서브_모듈_경로
  
  git submodule add 원격_저장소_URL 서브_모듈_경로
  ```
- 환경 변수 설정
  - 파일 열기
  ```
  sudo nano /etc/systemd/system/서비스_이름
  ```
  - 파일 수정
  ```
  [Unit]
  Description=...
  After=syslog.target
  
  [Service]
  User=ubuntu
  ExecStart=/usr/bin/java -jar 서비스_경로/소스_파일_경로
  Restart=always
  RestartSec=10
  
  // 환경 변수 설정
  Environment=AWS_MYSQL_URL=...
  Environment=AWS_MYSQL_PASSWORD=...
  Environment=AWS_POSTGRESQL_URL=...
  Environment=SMTP_PASSWORD=...
  Environment=AWS_S3_ACCESS_KEY=...
  Environment=AWS_S3_SECRET_KEY=...
  
  
  [Install]
  WantedBy=multi-user.target
  ```
  - 서비스 재시작
  ```
  sudo systemctl daemon-reload
  sudo systemctl restart 서비스_이름
  ```
  
참고 : https://velog.io/@ckdwns9121/posts
