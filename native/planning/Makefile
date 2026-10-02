CXX := xcrun clang++
BUILD_DIR ?= build
empty :=
space := $(empty) $(empty)
BUILD_TARGET_DIR := $(subst $(space),\$(space),$(BUILD_DIR))
.PHONY: test
test: $(BUILD_TARGET_DIR)/test-admission
	"$(BUILD_DIR)/test-admission"
$(BUILD_TARGET_DIR):
	mkdir -p "$@"
$(BUILD_TARGET_DIR)/test-admission: admission.cpp admission.h test_admission.cpp | $(BUILD_TARGET_DIR)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Werror -fsanitize=undefined admission.cpp test_admission.cpp -o "$@"
