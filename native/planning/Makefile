CXX := xcrun clang++
BUILD_DIR ?= build
.PHONY: test
test: $(BUILD_DIR)/test-admission
	"$(BUILD_DIR)/test-admission"
$(BUILD_DIR):
	mkdir -p "$@"
$(BUILD_DIR)/test-admission: admission.cpp admission.h test_admission.cpp | $(BUILD_DIR)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Werror -fsanitize=undefined admission.cpp test_admission.cpp -o "$@"
