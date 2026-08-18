#!/bin/bash
# Install dependencies and set up the dev environment.
# Run with: source scripts/install.sh

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

    # Check if uv is already installed
    if command -v uv >/dev/null 2>&1; then
        success "uv is already installed: $(uv --version)"
        return
    fi

    # Download and install uv
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        error "Failed to install uv."
        return 1
    fi

    # Add to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"

    # Verify installation
    if ! command -v uv >/dev/null 2>&1; then
        error "Failed to install uv."
        return 1
    fi

    success "uv installed successfully: $(uv --version)"
}

# Create and setup virtual environment
setup_venv() {
    log "Creating virtual environment..."

    # Remove existing venv if it exists
    if [ -d ".venv" ]; then
        warn "Removing existing virtual environment..."
        rm -rf .venv
    fi

    # Create new virtual environment
    if ! uv venv --python python3.13; then
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

# Main installation process: uv, venv, project dependencies.
# Uses `return` throughout (never `exit`) because this script is sourced —
# `exit` would terminate the user's shell.
main() {
    echo "Starting Installation"
    echo "=================================="

    log "Installing packages..."
    install_uv || return 1
    setup_venv || return 1

    echo ""
    echo "Installation complete!"
}

# Run main function
main "$@"
