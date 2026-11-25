Option Explicit
Dim fso, shell, net, procEnv, rootDir, pyw, py, stampPath, expectedId, stampContent, cmd, checkCode
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
Set net = CreateObject("WScript.Network")
Set procEnv = shell.Environment("PROCESS")

rootDir = fso.GetParentFolderName(WScript.ScriptFullName)
If Right(rootDir, 1) <> "\" Then
    expectedId = UCase(net.ComputerName & "::" & rootDir & "\::READY")
Else
    expectedId = UCase(net.ComputerName & "::" & rootDir & "::READY")
End If

procEnv("PYTHONUTF8") = "1"
procEnv("OPENCV_LOG_LEVEL") = "ERROR"

pyw = fso.BuildPath(rootDir, ".venv\Scripts\pythonw.exe")
py = fso.BuildPath(rootDir, ".venv\Scripts\python.exe")
stampPath = fso.BuildPath(rootDir, ".venv\.shortly_ready")

shell.CurrentDirectory = rootDir
If fso.FileExists(pyw) And fso.FileExists(py) Then
    If fso.FileExists(stampPath) Then
        On Error Resume Next
        stampContent = UCase(Trim(fso.OpenTextFile(stampPath, 1).ReadLine()))
        On Error GoTo 0
        If stampContent = expectedId Then
            cmd = """" & pyw & """ -O """ & fso.BuildPath(rootDir, "gui.py") & """"
            shell.Run cmd, 0, False
            WScript.Quit 0
        End If
    End If

    checkCode = shell.Run("""" & py & """ -c ""import importlib.util; mods=('yaml','pydantic','faster_whisper','cv2','yt_dlp','PIL','rich','numpy'); raise SystemExit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)""", 0, True)
    If checkCode = 0 Then
        On Error Resume Next
        fso.CreateTextFile(stampPath, True).WriteLine(expectedId)
        On Error GoTo 0
        cmd = """" & pyw & """ -O """ & fso.BuildPath(rootDir, "gui.py") & """"
        shell.Run cmd, 0, False
        WScript.Quit 0
    End If
End If

shell.Run "cmd /c """ & fso.BuildPath(rootDir, "Start Shortly.bat") & """", 1, False
