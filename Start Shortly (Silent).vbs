Option Explicit
Dim fso, shell, rootDir, pyw, py, cmd, checkCode
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

rootDir = fso.GetParentFolderName(WScript.ScriptFullName)
pyw = fso.BuildPath(rootDir, ".venv\Scripts\pythonw.exe")
py = fso.BuildPath(rootDir, ".venv\Scripts\python.exe")

shell.CurrentDirectory = rootDir
If fso.FileExists(pyw) And fso.FileExists(py) Then
    checkCode = shell.Run("""" & py & """ -c ""import importlib.util, sys; mods=('yaml','pydantic','faster_whisper','cv2','yt_dlp','mediapipe','PIL','rich'); raise SystemExit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)""", 0, True)
    If checkCode = 0 Then
        cmd = """" & pyw & """ """ & fso.BuildPath(rootDir, "gui.py") & """"
        shell.Run cmd, 0, False
        WScript.Quit 0
    End If
End If

shell.Run "cmd /c """ & fso.BuildPath(rootDir, "Start Shortly.bat") & """", 1, False
