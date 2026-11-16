"""Count Python executables, not names appearing in wrapper arguments."""
from pathlib import Path
import re

def python_workers(top):
    result=[]
    for line in top.splitlines():
        fields=line.split(None,3)
        if len(fields)!=4 or not fields[0].isdigit() or not fields[3].split():continue
        executable=Path(fields[3].split()[0]).name.strip('[]')
        if re.fullmatch(r'python(?:[0-9]+(?:\.[0-9]+)*)?',executable):
            result.append({'pid':int(fields[0]),'ppid':int(fields[1]),'comm':fields[2],'executable':executable})
    return result
