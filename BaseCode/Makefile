# GEMM ベース実装用Makefile

CC = gcc
CFLAGS = -O2 -march=native -std=c99 -Wall
LDFLAGS = -lm

# ターゲット
TARGET = gemm_base

# ソースファイル
SRC = gemm_base.c

# オブジェクトファイル
OBJ = $(SRC:.c=.o)

# デフォルトターゲット
all: $(TARGET)

# 実行ファイルの生成
$(TARGET): $(OBJ)
	$(CC) $(OBJ) $(LDFLAGS) -o $@

# オブジェクトファイルの生成
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# クリーン
clean:
	rm -f $(OBJ) $(TARGET)

# 実行（デフォルトサイズ）
run: $(TARGET)
	./$(TARGET)

# 小サイズでのテスト実行
test: $(TARGET)
	./$(TARGET) 512 512 512

# 性能評価用の各種サイズ実行
benchmark: $(TARGET)
	@echo "=== Small matrices ==="
	./$(TARGET) 512 512 512
	@echo ""
	@echo "=== Medium matrices ==="
	./$(TARGET) 1024 1024 1024
	@echo ""
	@echo "=== Large matrices ==="
	./$(TARGET) 2048 2048 2048

.PHONY: all clean run test benchmark