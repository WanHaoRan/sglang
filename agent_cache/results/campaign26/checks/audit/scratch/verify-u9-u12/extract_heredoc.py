"""Read-only: write the heredoc body of the Bash tool call at a transcript line index to a file."""
import json, sys
path, idx, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(path, errors="replace") as f:
    for i, line in enumerate(f):
        if i == idx:
            d = json.loads(line)
            for c in d["message"]["content"]:
                if c.get("type") == "tool_use":
                    cmd = c["input"]["command"]
                    start = cmd.index("<<'EOF'")
                    body = cmd[cmd.index("\n", start) + 1:]
                    body = body[:body.rindex("\nEOF")] + "\n"
                    open(dst, "w").write(body)
                    print("wrote", dst, len(body), "bytes; first command line:", cmd[:cmd.index("\n")])
            break
