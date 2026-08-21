CLANG ?= xcrun clang
BUILD_DIR ?= build
TARGET := $(BUILD_DIR)/strata-ane-probe
SOURCE := strata_ane_probe.m

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	mkdir -p $(BUILD_DIR)
	$(CLANG) -fobjc-arc -Wall -Wextra -Werror -O2 \
		-framework Foundation -framework IOSurface -o $@ $<

clean:
	rm -f $(TARGET)
