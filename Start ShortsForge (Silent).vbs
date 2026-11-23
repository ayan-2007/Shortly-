Option Explicit
Dim fso, shell, rootDir, pyw, py, cmd
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

rootDir = fso.GetParentFolderName(WScript.ScriptFullName)
pyw = fso.BuildPath(rootDir, ".venv\Scripts\pythonw.exe")
py = fso.BuildPath(rootDir, ".venv\Scripts\python.exe")

shell.CurrentDirectory = rootDir
If fso.FileExists(pyw) Then
    cmd = """" & pyw & """ """ & fso.BuildPath(rootDir, "gui.py") & """"
    shell.Run cmd, 0, False
ElseIf fso.FileExists(py) Then
    cmd = """" & py & """ """ & fso.BuildPath(rootDir, "gui.py") & """"
    shell.Run cmd, 0, False
Else
    shell.Run "cmd /c ""Start ShortsForge.bat""", 1, False
End If
