// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: dataset.proto

#include "dataset.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

namespace protoDataset {
class DatasetDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<Dataset> _instance;
} _Dataset_default_instance_;
}  // namespace protoDataset
static void InitDefaultsDataset_dataset_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::protoDataset::_Dataset_default_instance_;
    new (ptr) ::protoDataset::Dataset();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::protoDataset::Dataset::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<0> scc_info_Dataset_dataset_2eproto =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 0, InitDefaultsDataset_dataset_2eproto}, {}};

void InitDefaults_dataset_2eproto() {
  ::google::protobuf::internal::InitSCC(&scc_info_Dataset_dataset_2eproto.base);
}

::google::protobuf::Metadata file_level_metadata_dataset_2eproto[1];
constexpr ::google::protobuf::EnumDescriptor const** file_level_enum_descriptors_dataset_2eproto = nullptr;
constexpr ::google::protobuf::ServiceDescriptor const** file_level_service_descriptors_dataset_2eproto = nullptr;

const ::google::protobuf::uint32 TableStruct_dataset_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::protoDataset::Dataset, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::protoDataset::Dataset, numrows_),
  PROTOBUF_FIELD_OFFSET(::protoDataset::Dataset, numcols_),
  PROTOBUF_FIELD_OFFSET(::protoDataset::Dataset, data_),
  PROTOBUF_FIELD_OFFSET(::protoDataset::Dataset, labels_),
  PROTOBUF_FIELD_OFFSET(::protoDataset::Dataset, numlabels_),
};
static const ::google::protobuf::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::protoDataset::Dataset)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&::protoDataset::_Dataset_default_instance_),
};

::google::protobuf::internal::AssignDescriptorsTable assign_descriptors_table_dataset_2eproto = {
  {}, AddDescriptors_dataset_2eproto, "dataset.proto", schemas,
  file_default_instances, TableStruct_dataset_2eproto::offsets,
  file_level_metadata_dataset_2eproto, 1, file_level_enum_descriptors_dataset_2eproto, file_level_service_descriptors_dataset_2eproto,
};

const char descriptor_table_protodef_dataset_2eproto[] =
  "\n\rdataset.proto\022\014protoDataset\"`\n\007Dataset"
  "\022\017\n\007numRows\030\001 \001(\r\022\017\n\007numCols\030\002 \001(\r\022\020\n\004da"
  "ta\030\003 \003(\002B\002\020\001\022\016\n\006labels\030\004 \003(\005\022\021\n\tnumLabel"
  "s\030\005 \001(\rb\006proto3"
  ;
::google::protobuf::internal::DescriptorTable descriptor_table_dataset_2eproto = {
  false, InitDefaults_dataset_2eproto, 
  descriptor_table_protodef_dataset_2eproto,
  "dataset.proto", &assign_descriptors_table_dataset_2eproto, 135,
};

void AddDescriptors_dataset_2eproto() {
  static constexpr ::google::protobuf::internal::InitFunc deps[1] =
  {
  };
 ::google::protobuf::internal::AddDescriptors(&descriptor_table_dataset_2eproto, deps, 0);
}

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_dataset_2eproto = []() { AddDescriptors_dataset_2eproto(); return true; }();
namespace protoDataset {

// ===================================================================

void Dataset::InitAsDefaultInstance() {
}
class Dataset::HasBitSetters {
 public:
};

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int Dataset::kNumRowsFieldNumber;
const int Dataset::kNumColsFieldNumber;
const int Dataset::kDataFieldNumber;
const int Dataset::kLabelsFieldNumber;
const int Dataset::kNumLabelsFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

Dataset::Dataset()
  : ::google::protobuf::Message(), _internal_metadata_(nullptr) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:protoDataset.Dataset)
}
Dataset::Dataset(const Dataset& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(nullptr),
      data_(from.data_),
      labels_(from.labels_) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::memcpy(&numrows_, &from.numrows_,
    static_cast<size_t>(reinterpret_cast<char*>(&numlabels_) -
    reinterpret_cast<char*>(&numrows_)) + sizeof(numlabels_));
  // @@protoc_insertion_point(copy_constructor:protoDataset.Dataset)
}

void Dataset::SharedCtor() {
  ::memset(&numrows_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&numlabels_) -
      reinterpret_cast<char*>(&numrows_)) + sizeof(numlabels_));
}

Dataset::~Dataset() {
  // @@protoc_insertion_point(destructor:protoDataset.Dataset)
  SharedDtor();
}

void Dataset::SharedDtor() {
}

void Dataset::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const Dataset& Dataset::default_instance() {
  ::google::protobuf::internal::InitSCC(&::scc_info_Dataset_dataset_2eproto.base);
  return *internal_default_instance();
}


void Dataset::Clear() {
// @@protoc_insertion_point(message_clear_start:protoDataset.Dataset)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  data_.Clear();
  labels_.Clear();
  ::memset(&numrows_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&numlabels_) -
      reinterpret_cast<char*>(&numrows_)) + sizeof(numlabels_));
  _internal_metadata_.Clear();
}

#if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
const char* Dataset::_InternalParse(const char* begin, const char* end, void* object,
                  ::google::protobuf::internal::ParseContext* ctx) {
  auto msg = static_cast<Dataset*>(object);
  ::google::protobuf::int32 size; (void)size;
  int depth; (void)depth;
  ::google::protobuf::uint32 tag;
  ::google::protobuf::internal::ParseFunc parser_till_end; (void)parser_till_end;
  auto ptr = begin;
  while (ptr < end) {
    ptr = ::google::protobuf::io::Parse32(ptr, &tag);
    GOOGLE_PROTOBUF_PARSER_ASSERT(ptr);
    switch (tag >> 3) {
      // uint32 numRows = 1;
      case 1: {
        if (static_cast<::google::protobuf::uint8>(tag) != 8) goto handle_unusual;
        msg->set_numrows(::google::protobuf::internal::ReadVarint(&ptr));
        GOOGLE_PROTOBUF_PARSER_ASSERT(ptr);
        break;
      }
      // uint32 numCols = 2;
      case 2: {
        if (static_cast<::google::protobuf::uint8>(tag) != 16) goto handle_unusual;
        msg->set_numcols(::google::protobuf::internal::ReadVarint(&ptr));
        GOOGLE_PROTOBUF_PARSER_ASSERT(ptr);
        break;
      }
      // repeated float data = 3 [packed = true];
      case 3: {
        if (static_cast<::google::protobuf::uint8>(tag) == 26) {
          ptr = ::google::protobuf::io::ReadSize(ptr, &size);
          GOOGLE_PROTOBUF_PARSER_ASSERT(ptr);
          parser_till_end = ::google::protobuf::internal::PackedFloatParser;
          object = msg->mutable_data();
          if (size > end - ptr) goto len_delim_till_end;
          auto newend = ptr + size;
          if (size) ptr = parser_till_end(ptr, newend, object, ctx);
          GOOGLE_PROTOBUF_PARSER_ASSERT(ptr == newend);
          break;
        } else if (static_cast<::google::protobuf::uint8>(tag) != 29) goto handle_unusual;
        do {
          msg->add_data(::google::protobuf::io::UnalignedLoad<float>(ptr));
          ptr += sizeof(float);
          if (ptr >= end) break;
        } while ((::google::protobuf::io::UnalignedLoad<::google::protobuf::uint64>(ptr) & 255) == 29 && (ptr += 1));
        break;
      }
      // repeated int32 labels = 4;
      case 4: {
        if (static_cast<::google::protobuf::uint8>(tag) == 34) {
          ptr = ::google::protobuf::io::ReadSize(ptr, &size);
          GOOGLE_PROTOBUF_PARSER_ASSERT(ptr);
          parser_till_end = ::google::protobuf::internal::PackedInt32Parser;
          object = msg->mutable_labels();
          if (size > end - ptr) goto len_delim_till_end;
          auto newend = ptr + size;
          if (size) ptr = parser_till_end(ptr, newend, object, ctx);
          GOOGLE_PROTOBUF_PARSER_ASSERT(ptr == newend);
          break;
        } else if (static_cast<::google::protobuf::uint8>(tag) != 32) goto handle_unusual;
        do {
          msg->add_labels(::google::protobuf::internal::ReadVarint(&ptr));
          GOOGLE_PROTOBUF_PARSER_ASSERT(ptr);
          if (ptr >= end) break;
        } while ((::google::protobuf::io::UnalignedLoad<::google::protobuf::uint64>(ptr) & 255) == 32 && (ptr += 1));
        break;
      }
      // uint32 numLabels = 5;
      case 5: {
        if (static_cast<::google::protobuf::uint8>(tag) != 40) goto handle_unusual;
        msg->set_numlabels(::google::protobuf::internal::ReadVarint(&ptr));
        GOOGLE_PROTOBUF_PARSER_ASSERT(ptr);
        break;
      }
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->EndGroup(tag);
          return ptr;
        }
        auto res = UnknownFieldParse(tag, {_InternalParse, msg},
          ptr, end, msg->_internal_metadata_.mutable_unknown_fields(), ctx);
        ptr = res.first;
        GOOGLE_PROTOBUF_PARSER_ASSERT(ptr != nullptr);
        if (res.second) return ptr;
      }
    }  // switch
  }  // while
  return ptr;
len_delim_till_end:
  return ctx->StoreAndTailCall(ptr, end, {_InternalParse, msg},
                               {parser_till_end, object}, size);
}
#else  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
bool Dataset::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!PROTOBUF_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:protoDataset.Dataset)
  for (;;) {
    ::std::pair<::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // uint32 numRows = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) == (8 & 0xFF)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint32, ::google::protobuf::internal::WireFormatLite::TYPE_UINT32>(
                 input, &numrows_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // uint32 numCols = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) == (16 & 0xFF)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint32, ::google::protobuf::internal::WireFormatLite::TYPE_UINT32>(
                 input, &numcols_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // repeated float data = 3 [packed = true];
      case 3: {
        if (static_cast< ::google::protobuf::uint8>(tag) == (26 & 0xFF)) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadPackedPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, this->mutable_data())));
        } else if (static_cast< ::google::protobuf::uint8>(tag) == (29 & 0xFF)) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadRepeatedPrimitiveNoInline<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 1, 26u, input, this->mutable_data())));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // repeated int32 labels = 4;
      case 4: {
        if (static_cast< ::google::protobuf::uint8>(tag) == (34 & 0xFF)) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadPackedPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, this->mutable_labels())));
        } else if (static_cast< ::google::protobuf::uint8>(tag) == (32 & 0xFF)) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadRepeatedPrimitiveNoInline<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 1, 34u, input, this->mutable_labels())));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // uint32 numLabels = 5;
      case 5: {
        if (static_cast< ::google::protobuf::uint8>(tag) == (40 & 0xFF)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint32, ::google::protobuf::internal::WireFormatLite::TYPE_UINT32>(
                 input, &numlabels_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:protoDataset.Dataset)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:protoDataset.Dataset)
  return false;
#undef DO_
}
#endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER

void Dataset::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:protoDataset.Dataset)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // uint32 numRows = 1;
  if (this->numrows() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt32(1, this->numrows(), output);
  }

  // uint32 numCols = 2;
  if (this->numcols() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt32(2, this->numcols(), output);
  }

  // repeated float data = 3 [packed = true];
  if (this->data_size() > 0) {
    ::google::protobuf::internal::WireFormatLite::WriteTag(3, ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED, output);
    output->WriteVarint32(_data_cached_byte_size_.load(
        std::memory_order_relaxed));
    ::google::protobuf::internal::WireFormatLite::WriteFloatArray(
      this->data().data(), this->data_size(), output);
  }

  // repeated int32 labels = 4;
  if (this->labels_size() > 0) {
    ::google::protobuf::internal::WireFormatLite::WriteTag(4, ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED, output);
    output->WriteVarint32(_labels_cached_byte_size_.load(
        std::memory_order_relaxed));
  }
  for (int i = 0, n = this->labels_size(); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32NoTag(
      this->labels(i), output);
  }

  // uint32 numLabels = 5;
  if (this->numlabels() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt32(5, this->numlabels(), output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        _internal_metadata_.unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:protoDataset.Dataset)
}

::google::protobuf::uint8* Dataset::InternalSerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:protoDataset.Dataset)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // uint32 numRows = 1;
  if (this->numrows() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt32ToArray(1, this->numrows(), target);
  }

  // uint32 numCols = 2;
  if (this->numcols() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt32ToArray(2, this->numcols(), target);
  }

  // repeated float data = 3 [packed = true];
  if (this->data_size() > 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteTagToArray(
      3,
      ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
      target);
    target = ::google::protobuf::io::CodedOutputStream::WriteVarint32ToArray(
        _data_cached_byte_size_.load(std::memory_order_relaxed),
         target);
    target = ::google::protobuf::internal::WireFormatLite::
      WriteFloatNoTagToArray(this->data_, target);
  }

  // repeated int32 labels = 4;
  if (this->labels_size() > 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteTagToArray(
      4,
      ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
      target);
    target = ::google::protobuf::io::CodedOutputStream::WriteVarint32ToArray(
        _labels_cached_byte_size_.load(std::memory_order_relaxed),
         target);
    target = ::google::protobuf::internal::WireFormatLite::
      WriteInt32NoTagToArray(this->labels_, target);
  }

  // uint32 numLabels = 5;
  if (this->numlabels() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt32ToArray(5, this->numlabels(), target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:protoDataset.Dataset)
  return target;
}

size_t Dataset::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:protoDataset.Dataset)
  size_t total_size = 0;

  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        _internal_metadata_.unknown_fields());
  }
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated float data = 3 [packed = true];
  {
    unsigned int count = static_cast<unsigned int>(this->data_size());
    size_t data_size = 4UL * count;
    if (data_size > 0) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(
            static_cast<::google::protobuf::int32>(data_size));
    }
    int cached_size = ::google::protobuf::internal::ToCachedSize(data_size);
    _data_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  // repeated int32 labels = 4;
  {
    size_t data_size = ::google::protobuf::internal::WireFormatLite::
      Int32Size(this->labels_);
    if (data_size > 0) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(
            static_cast<::google::protobuf::int32>(data_size));
    }
    int cached_size = ::google::protobuf::internal::ToCachedSize(data_size);
    _labels_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  // uint32 numRows = 1;
  if (this->numrows() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::UInt32Size(
        this->numrows());
  }

  // uint32 numCols = 2;
  if (this->numcols() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::UInt32Size(
        this->numcols());
  }

  // uint32 numLabels = 5;
  if (this->numlabels() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::UInt32Size(
        this->numlabels());
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void Dataset::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:protoDataset.Dataset)
  GOOGLE_DCHECK_NE(&from, this);
  const Dataset* source =
      ::google::protobuf::DynamicCastToGenerated<Dataset>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:protoDataset.Dataset)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:protoDataset.Dataset)
    MergeFrom(*source);
  }
}

void Dataset::MergeFrom(const Dataset& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:protoDataset.Dataset)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  data_.MergeFrom(from.data_);
  labels_.MergeFrom(from.labels_);
  if (from.numrows() != 0) {
    set_numrows(from.numrows());
  }
  if (from.numcols() != 0) {
    set_numcols(from.numcols());
  }
  if (from.numlabels() != 0) {
    set_numlabels(from.numlabels());
  }
}

void Dataset::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:protoDataset.Dataset)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Dataset::CopyFrom(const Dataset& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:protoDataset.Dataset)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Dataset::IsInitialized() const {
  return true;
}

void Dataset::Swap(Dataset* other) {
  if (other == this) return;
  InternalSwap(other);
}
void Dataset::InternalSwap(Dataset* other) {
  using std::swap;
  _internal_metadata_.Swap(&other->_internal_metadata_);
  data_.InternalSwap(&other->data_);
  labels_.InternalSwap(&other->labels_);
  swap(numrows_, other->numrows_);
  swap(numcols_, other->numcols_);
  swap(numlabels_, other->numlabels_);
}

::google::protobuf::Metadata Dataset::GetMetadata() const {
  ::google::protobuf::internal::AssignDescriptors(&::assign_descriptors_table_dataset_2eproto);
  return ::file_level_metadata_dataset_2eproto[kIndexInFileMessages];
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace protoDataset
namespace google {
namespace protobuf {
template<> PROTOBUF_NOINLINE ::protoDataset::Dataset* Arena::CreateMaybeMessage< ::protoDataset::Dataset >(Arena* arena) {
  return Arena::CreateInternal< ::protoDataset::Dataset >(arena);
}
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>