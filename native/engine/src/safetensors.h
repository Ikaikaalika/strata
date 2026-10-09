// Read-only, memory-mapped safetensors checkpoints (single file or shards).
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace lokahi {

enum class DType { F32, F16, BF16, U32, I32, U16, U8, I8, I64, U64, F64, Bool };

size_t dtype_size(DType dtype);
const char* dtype_name(DType dtype);

struct TensorView {
  std::string name;
  DType dtype = DType::F32;
  std::vector<int64_t> shape;
  const uint8_t* data = nullptr;  // Points into a read-only mapping.
  size_t nbytes = 0;

  int64_t numel() const;
  int64_t dim(size_t index) const;
  template <typename T>
  const T* as() const {
    return reinterpret_cast<const T*>(data);
  }
};

class MappedFile {
 public:
  explicit MappedFile(const std::string& path);
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
  uint8_t* data_ = nullptr;
  size_t size_ = 0;
};

class Checkpoint {
 public:
  // Opens every *.safetensors file in `directory`.
  static Checkpoint open_directory(const std::string& directory);

  const TensorView* find(const std::string& name) const;
  const TensorView& get(const std::string& name) const;
  bool contains(const std::string& name) const { return find(name) != nullptr; }
  const std::map<std::string, TensorView>& tensors() const { return tensors_; }
  size_t total_bytes() const;

 private:
  void add_file(const std::string& path);

  std::vector<std::shared_ptr<MappedFile>> files_;
  std::map<std::string, TensorView> tensors_;
};

std::string read_text_file(const std::string& path);

}  // namespace lokahi
