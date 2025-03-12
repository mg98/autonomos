# Default target
all:
	@echo "Run 'make clean' to clean temporary files"

install:
	git submodule update --init --recursive
	pip install -r requirements.txt

# Clean target to remove temporary files and directories
clean:
	@echo "Cleaning temporary files and directories..."
	rm -rf .tmp/ .cache out/ __pycache__/ */__pycache__/
	@echo "Clean complete."

cc:
	rm -f .cache
	@echo "Cache cleared."

.PHONY: all clean cc
