#!/data/data/com.termux/files/usr/bin/bash
#
# DOCUMENTACIÓN_AQUÍ
# Instalador de mini_ai C++20 para Termux.
# Instala las herramientas mínimas, configura CMake y compila los ejecutables.
# No instala Python, frameworks de IA ni modelos externos.
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"

say() {
    printf '\n[mini_ai] %s\n' "$1"
}

if ! command -v pkg >/dev/null 2>&1; then
    printf '[ERROR] Este script debe ejecutarse dentro de Termux.\n' >&2
    exit 1
fi

say "Actualizando paquetes de Termux"
pkg update -y

say "Instalando clang, CMake y Ninja"
pkg install -y clang cmake ninja

say "Configurando CMake"
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON

say "Compilando mini_ai"
cmake --build "$BUILD_DIR" --parallel "$(nproc 2>/dev/null || echo 1)"

say "Ejecutando pruebas"
ctest --test-dir "$BUILD_DIR" --output-on-failure

printf '\n[PASS] mini_ai quedó instalado y compilado.\n'
printf 'Ejecutables: %s/mini_ai_train y %s/mini_ai_test\n' "$BUILD_DIR" "$BUILD_DIR"
printf 'Usa "%s/mini_ai_train --help" para ver las opciones.\n' "$BUILD_DIR"
