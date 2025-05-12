# Default target
all:
	@echo "Run 'make clean' to clean temporary files"

install:
	git submodule update --init --recursive
	pip install -r requirements.txt

clean:
	@echo "Cleaning temporary files and directories..."
	rm -rf .tmp/ .cache out/ __pycache__/ */__pycache__/
	@echo "Clean complete."

.PHONY: all clean
