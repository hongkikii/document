# 로그 확인 명령어
```
sudo journalctl -u JoA.service
```

## 옵션

1. **`-since=TIME`**: 특정 시간 이후의 로그를 표시

    ```
    sudo journalctl -u JoA.service --since "2024-01-30 12:00:00"
    ```

    ```
    sudo journalctl -u JoA.service --since "1 hour ago"
    ```

2. **`-until=TIME`**: 특정 시간 이전의 로그를 표시

    ```
    sudo journalctl -u JoA.service --until "2024-01-30 18:00:00"
    ```

3. **`n, --lines=NUM`**: 최근 몇 줄의 로그를 표시

    ```
    sudo journalctl -u JoA.service -n 20
    ```

   위 명령어는 최근 20줄의 로그 표시

4. **`f, --follow`**: 로그를 실시간으로 모니터링

    ```
    sudo journalctl -u JoA.service -f
    ```

   아래와 같이 옵션을 함께 사용할 수도 있음

    ```
    sudo journalctl -u JoA.service -n 50 -f
    ```

5. **`r, --reverse`**: 로그 역순으로 표시 (가장 최근 로그가 먼저 표시)

    ```
    sudo journalctl -u JoA.service -r
    ```

6. **`-no-pager`**: 로그를 페이저에 표시하지 않고 직접 터미널에 출력

7. **`o, --output=FORMAT`**: 출력 형식을 설정 (예: **`short`**, **`json`**, **`cat`** 등)

8. **`-grep=PATTERN`**: 로그에서 특정 패턴을 검색

    ```
    sudo journalctl --grep "error"
    ```

9. **`-disk-usage`**: 로그 저장 공간 사용량을 표시

# 시스템 상태 확인 명령어

```
systemctl status JoA.service
```

종합적으로 간단한 상태 확인 가능
