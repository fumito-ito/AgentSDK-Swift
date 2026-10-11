# Common developer commands for AgentSDK-Swift.
#
# These targets work both with a full Xcode installation and with the
# Command Line Tools only. When only the Command Line Tools are installed,
# `swift test` cannot find the `Testing` module and SwiftLint cannot find
# SourceKit, so the required search paths are injected automatically.

SWIFT ?= swift
DEVELOPER_DIR_PATH := $(shell xcode-select -p 2>/dev/null)
UPSTREAM_DIR ?= .vendor/openai-agents-python
UPSTREAM_REPO ?= https://github.com/openai/openai-agents-python

ifeq ($(DEVELOPER_DIR_PATH),/Library/Developer/CommandLineTools)
CLT_FRAMEWORKS := $(DEVELOPER_DIR_PATH)/Library/Developer/Frameworks
CLT_LIBS := $(DEVELOPER_DIR_PATH)/Library/Developer/usr/lib
TEST_FLAGS := -Xswiftc -F -Xswiftc $(CLT_FRAMEWORKS) \
	-Xlinker -F -Xlinker $(CLT_FRAMEWORKS) \
	-Xlinker -rpath -Xlinker $(CLT_FRAMEWORKS) \
	-Xlinker -rpath -Xlinker $(CLT_LIBS)
LINT_ENV := TOOLCHAIN_DIR=$(DEVELOPER_DIR_PATH)
endif

# Prefer the SwiftLint version pinned in mise.toml when mise is available.
SWIFTLINT := $(shell command -v mise >/dev/null 2>&1 && echo "mise exec -- swiftlint" || echo swiftlint)

.PHONY: help bootstrap build test lint lint-fix format check clean upstream upstream-version

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

bootstrap: ## Install pinned developer tools and resolve packages.
	mise install
	$(SWIFT) package resolve

build: ## Build all targets.
	$(SWIFT) build

test: ## Run the test suite. Pass FILTER=<pattern> to run a subset.
	$(SWIFT) test $(TEST_FLAGS) $(if $(FILTER),--filter $(FILTER),)

lint: ## Run SwiftLint in strict mode.
	$(LINT_ENV) $(SWIFTLINT) lint --strict --quiet

lint-fix: ## Auto-correct SwiftLint violations where possible.
	$(LINT_ENV) $(SWIFTLINT) lint --fix --quiet

check: lint build test ## Run lint, build, and tests (the full local verification stack).

clean: ## Remove build products.
	$(SWIFT) package clean

upstream: ## Check out the tracked upstream Python SDK release into .vendor/. Pass VERSION=<x.y.z> to override.
	./scripts/upstream/checkout.sh $(VERSION)

upstream-version: ## Print the upstream Python SDK version this port tracks.
	@cat UPSTREAM_VERSION
