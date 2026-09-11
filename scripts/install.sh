#!/bin/bash
# Install dependencies and set up the dev environment.
# Run with: source scripts/install.sh
#
# Uses `return` throughout (never `exit`) because this script is sourced —
# `exit` would terminate the user's shell.

log() {
    echo "[INFO] $1"
}

warn() {
    echo "[WARN] $1"
}

error() {
    echo "[ERROR] $1"
}

success() {
    echo "[SUCCESS] $1"
}

# Install uv package manager
install_uv() {
    log "Installing uv package manager..."

    if command -v uv >/dev/null 2>&1; then
        success "uv is already installed: $(uv --version)"
        return
    fi

    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        error "Failed to install uv."
        return 1
    fi

    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv >/dev/null 2>&1; then
        error "Failed to install uv."
        return 1
    fi

    success "uv installed successfully: $(uv --version)"
}

# Create and setup virtual environment
setup_venv() {
    log "Creating virtual environment..."

    if [ -d ".venv" ]; then
        warn "Removing existing virtual environment..."
        rm -rf .venv
    fi

    # Free-threaded 3.14t so GroupEnv max_threads can run env steps in parallel.
    log "Ensuring free-threaded Python 3.14t is available..."
    if ! uv python install 3.14t; then
        error "Failed to install Python 3.14t"
        return 1
    fi

    if ! uv venv --python 3.14t; then
        error "Failed to create virtual environment"
        return 1
    fi

    success "Virtual environment created"

    # Install project dependencies (core + all optional extras).
    log "Installing project dependencies..."
    if ! uv pip install -e ".[dev,all]" --python .venv/bin/python; then
        error "Failed to install project dependencies"
        return 1
    fi

    success "Project dependencies installed"
}

# Main installation process: uv, venv, project dependencies
main() {
    cd "$(dirname "${BASH_SOURCE[0]}")/.." || return 1

    echo "Starting Installation"
    echo "=================================="

    log "Installing packages..."
    install_uv || return 1
    setup_venv || return 1

    echo ""
    echo "Installation complete!"
    echo ""
    log "Activate the virtual environment:"
    echo "  source .venv/bin/activate"
}

main "$@"
