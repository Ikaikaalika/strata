#include "safetensors.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>

#include "common.h"
#include "json.h"

namespace lokahi {

size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::F64:
    case DType::I64:
    case DType::U64:
      return 8;
    case DType::F32:
    case DType::U32:
    case DType::I32:
      return 4;
    case DType::F16:
    case DType::BF16:
    case DType::U16:
      return 2;
    case DType::U8:
    case DType::I8:
    case DType::Bool:
      return 1;
  }
  return 0;
}

const char* dtype_name(DType dtype) {
  switch (dtype) {
    case DType::F64: return "F64";
    case DType::F32: return "F32";
    case DType::F16: return "F16";
    case DType::BF16: return "BF16";
    case DType::I64: return "I64";
    case DType::U64: return "U64";
    case DType::I32: return "I32";
    case DType::U32: return "U32";
    case DType::U16: return "U16";
    case DType::U8: return "U8";
    case DType::I8: return "I8";
    case DType::Bool: return "BOOL";
  }
  return "?";
}

static DType parse_dtype(const std::string& name) {
  static const std::map<std::string, DType> table = {
      {"F64", DType::F64}, {"F32", DType::F32},   {"F16", DType::F16}, {"BF16", DType::BF16},
      {"I64", DType::I64}, {"U64", DType::U64},   {"I32", DType::I32}, {"U32", DType::U32},
      {"U16", DType::U16}, {"U8", DType::U8},     {"I8", DType::I8},   {"BOOL", DType::Bool},
  };
  auto it = table.find(name);
  LK_CHECK(it != table.end(), "safetensors: unsupported dtype " + name);
  return it->second;
}

int64_t TensorView::numel() const {
  int64_t count = 1;
  for (int64_t d : shape) count *= d;
  return count;
}

int64_t TensorView::dim(size_t index) const {
  LK_CHECK(index < shape.size(), "tensor " + name + ": dimension out of range");
  return shape[index];
}

MappedFile::MappedFile(const std::string& path) : path_(path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  LK_CHECK(fd >= 0, "cannot open " + path + ": " + std::strerror(errno));
  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    fail("cannot stat " + path);
  }
  size_ = static_cast<size_t>(st.st_size);
  if (size_ > 0) {
    void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
      ::close(fd);
      fail("cannot mmap " + path + ": " + std::strerror(errno));
    }
    data_ = static_cast<uint8_t*>(mapped);
  }
  ::close(fd);
}

MappedFile::~MappedFile() {
  if (data_ != nullptr) ::munmap(data_, size_);
}

std::string read_text_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  LK_CHECK(in.good(), "cannot read " + path);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

void Checkpoint::add_file(const std::string& path) {
  auto file = std::make_shared<MappedFile>(path);
  LK_CHECK(file->size() >= 8, "safetensors: file too small: " + path);
  uint64_t header_len = 0;
  std::memcpy(&header_len, file->data(), 8);  // Little-endian on all supported hosts.
  LK_CHECK(header_len <= 100ull * 1024 * 1024 && 8 + header_len <= file->size(),
           "safetensors: invalid header length in " + path);
  std::string_view header_text(reinterpret_cast<const char*>(file->data() + 8), header_len);
  Json header = Json::parse(header_text);
  const uint8_t* payload = file->data() + 8 + header_len;
  size_t payload_size = file->size() - 8 - header_len;
  for (const auto& [name, entry] : header.as_object()) {
    if (name == "__metadata__") continue;
    TensorView view;
    view.name = name;
    view.dtype = parse_dtype(entry.at("dtype").as_string());
    for (const Json& dim : entry.at("shape").as_array()) view.shape.push_back(dim.as_int());
    const auto& offsets = entry.at("data_offsets").as_array();
    LK_CHECK(offsets.size() == 2, "safetensors: bad offsets for " + name);
    uint64_t begin = static_cast<uint64_t>(offsets[0].as_int());
    uint64_t end = static_cast<uint64_t>(offsets[1].as_int());
    LK_CHECK(begin <= end && end <= payload_size, "safetensors: offsets out of range for " + name);
    view.nbytes = static_cast<size_t>(end - begin);
    LK_CHECK(view.nbytes == static_cast<size_t>(view.numel()) * dtype_size(view.dtype),
             "safetensors: size mismatch for " + name);
    view.data = payload + begin;
    LK_CHECK(tensors_.find(name) == tensors_.end(), "safetensors: duplicate tensor " + name);
    tensors_.emplace(name, std::move(view));
  }
  files_.push_back(std::move(file));
}

Checkpoint Checkpoint::open_directory(const std::string& directory) {
  Checkpoint checkpoint;
  DIR* dir = ::opendir(directory.c_str());
  LK_CHECK(dir != nullptr, "cannot open model directory " + directory);
  std::vector<std::string> names;
  while (dirent* entry = ::readdir(dir)) {
    std::string name = entry->d_name;
    const std::string suffix = ".safetensors";
    if (name.size() > suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      names.push_back(name);
    }
  }
  ::closedir(dir);
  LK_CHECK(!names.empty(), "no .safetensors files in " + directory);
  std::sort(names.begin(), names.end());
  for (const std::string& name : names) checkpoint.add_file(directory + "/" + name);
  return checkpoint;
}

const TensorView* Checkpoint::find(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

const TensorView& Checkpoint::get(const std::string& name) const {
  const TensorView* view = find(name);
  LK_CHECK(view != nullptr, "checkpoint: missing tensor " + name);
  return *view;
}

size_t Checkpoint::total_bytes() const {
  size_t total = 0;
  for (const auto& [_, view] : tensors_) total += view.nbytes;
  return total;
}

}  // namespace lokahi
