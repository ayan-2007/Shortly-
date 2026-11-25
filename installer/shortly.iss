; Inno Setup script for Shortly Studio
#define MyAppName "Shortly Studio"
#define MyAppVersion "1.2.0"
#define MyAppPublisher "Shortly Studio"
#define MyAppExeName "ShortlyStudio.exe"

[Setup]
AppId={{8E9B7D4C-2F5A-4B1E-9C3D-6A1F8E2B4D90}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Shortly Studio
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\release
OutputBaseFilename=Shortly-Setup
SetupIconFile=shortly.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Standalone compiled app + bundled Python runtime + libraries + FFmpeg
Source: "..\dist\ShortlyStudio\*"; DestDir: "{app}"; Excludes: "config.yaml,.env,.env.example,shortly.log"; Flags: ignoreversion recursesubdirs createallsubdirs
; Keep user's config.yaml on upgrades
Source: "..\dist\ShortlyStudio\config.yaml"; DestDir: "{app}"; Flags: onlyifdoesntexist
Source: "..\dist\ShortlyStudio\.env.example"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
