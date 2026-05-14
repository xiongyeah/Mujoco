# 使用项目虚拟环境里的 Python（已安装 mujoco）
# 用法: .\py.ps1 scripts\verify_kinematics.py
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Args
)
$exe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $exe)) {
    Write-Host '未找到 .venv。请在 D:\Mujoco 下执行:' -ForegroundColor Yellow
    Write-Host '  python -m venv .venv' -ForegroundColor Yellow
    Write-Host '  .\.venv\Scripts\pip install -r requirements.txt' -ForegroundColor Yellow
    exit 1
}
& $exe @Args
