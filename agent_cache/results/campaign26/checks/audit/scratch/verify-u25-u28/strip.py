import sys, re, html
for p in sys.argv[1:]:
    s = open(p, 'r', errors='replace').read()
    s = re.sub(r'(?is)<(script|style)[^>]*>.*?</\1>', ' ', s)
    s = re.sub(r'(?i)<br\s*/?>|</(p|div|li|tr|h\d|td|th|section)>', '\n', s)
    s = re.sub(r'(?s)<[^>]+>', ' ', s)
    s = html.unescape(s)
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n\s*\n+', '\n', s)
    print('=====', p.split('/')[-1], len(s)); print(s.strip()[:6000])
