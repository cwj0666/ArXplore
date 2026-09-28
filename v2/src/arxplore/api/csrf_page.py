CSRF_PAGE_HEAD = (
    "<!DOCTYPE html>\n"
    '<html lang="en">\n'
    "<head>\n"
    '  <meta http-equiv="content-type" content="text/html; charset=utf-8">\n'
    '  <meta name="robots" content="NONE,NOARCHIVE">\n'
    "  <title>403 Forbidden</title>\n"
    "  <style>\n"
    "    html * { padding:0; margin:0; }\n"
    "    body * { padding:10px 20px; }\n"
    "    body * * { padding:0; }\n"
    "    body { font-family: sans-serif; background:#eee; color:#000; }\n"
    "    body>div { border-bottom:1px solid #ddd; }\n"
    "    h1 { font-weight:normal; margin-bottom:.4em; }\n"
    "    h1 span { font-size:60%; color:#666; font-weight:normal; }\n"
    "    #info { background:#f6f6f6; }\n"
    "    #info ul { margin: 0.5em 4em; }\n"
    "    #info p, #summary p { padding-top:10px; }\n"
    "    #summary { background: #ffc; }\n"
    "    #explanation { background:#eee; border-bottom: 0px none; }\n"
    "  </style>\n"
    "</head>\n"
    "<body>\n"
    '<div id="summary">\n'
    "  <h1>Forbidden <span>(403)</span></h1>\n"
    "  <p>CSRF 검증에 실패했습니다. 요청을 중단하였습니다.</p>\n"
    "\n"
    "\n"
)

CSRF_COOKIE_HELP = (
    "  <p>이 메세지가 보이는 이유는 사이트가 폼을 제출할 때 CSRF 쿠키를 필요로 하기 때문입니다. 이 쿠키는 보안상의 이유로 필요하며, 제3자에 의해 당신의 브라우저가 해킹당하고 있지 않다는 것을 보장합니다.</p>\n"
    "  <p>만약 브라우저 설정에서 쿠키를 비활성화 시켰을 경우, 적어도 이 사이트나 &#x27;동일-출처&#x27; 요청에 대해서는 활성화 시키십시오.</p>\n"
    "\n"
)

CSRF_PAGE_TAIL = (
    "</div>\n"
    "\n"
    '<div id="explanation">\n'
    "  <p><small>DEBUG=True 로 더 많은 정보를 확인할 수 있습니다.</small></p>\n"
    "</div>\n"
    "\n"
    "</body>\n"
    "</html>\n"
)


def csrf_failure_page(*, missing_cookie: bool) -> str:
    return CSRF_PAGE_HEAD + (CSRF_COOKIE_HELP if missing_cookie else "") + CSRF_PAGE_TAIL
