#!/data/data/com.termux/files/usr/bin/bash
#
# DOCUMENTACIÓN_AQUÍ
# Instalador remoto/local de mini_ai C++20 para Termux.
# Puede ejecutarse directamente desde el repositorio o mediante:
# curl -fsSL <URL_RAW> | bash
# En modo remoto descarga la rama cpp20-rewrite y vuelve a ejecutarse localmente.
#
set -euo pipefail

REPO_URL="https://github.com/CRISTOP-bot/mini_ai"
BRANCH="cpp20-rewrite"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || true)"

say() { printf '\n[mini_ai] %s\n' "$1"; }

if [[ ! -f "${SCRIPT_DIR}/CMakeLists.txt" ]]; then
    if ! command -v pkg >/dev/null 2>&1; then
        printf '[ERROR] Ejecuta este instalador dentro de Termux.\n' >&2
        exit 1
    fi

    say "Instalación remota: preparando Git"
    pkg update -y
    pkg install -y git

    TARGET="${HOME}/mini_ai_cpp"
    if [[ -e "$TARGET" ]]; then
        TARGET="${HOME}/mini_ai_cpp_$(date +%Y%m%d_%H%M%S)"
    fi

    say "Descargando mini_ai (${BRANCH})"
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$TARGET"
    exec bash "$TARGET/install.sh" "$@"
fi

PROJECT_DIR="$SCRIPT_DIR"
BUILD_DIR="${PROJECT_DIR}/build"

if ! command -v pkg >/dev/null 2>&1; then
    printf '[ERROR] Este script debe ejecutarse dentro de Termux.\n' >&2
    exit 1
fi

say "Actualizando paquetes de Termux"
pkg update -y
say "Instalando clang, CMake, Ninja y Git"
pkg install -y clang cmake ninja git
say "Configurando CMake"
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON
say "Compilando mini_ai"
cmake --build "$BUILD_DIR" --parallel "$(nproc 2>/dev/null || echo 1)"
say "Ejecutando pruebas"
ctest --test-dir "$BUILD_DIR" --output-on-failure
printf '\n[PASS] mini_ai quedó instalado y compilado.\n'
printf 'Proyecto: %s\n' "$PROJECT_DIR"
printf 'Ejecutables: %s/mini_ai_train y %s/mini_ai_test\n' "$BUILD_DIR" "$BUILD_DIR"
