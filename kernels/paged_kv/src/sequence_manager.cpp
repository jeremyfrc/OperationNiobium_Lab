#include "sequence_manager.h"
#include "check.h"

namespace paged_kv {

bool append_kv(PagedKVCache& cache, BlockTable& table, int layer, const float* kNew, const float* vNew, int nNew) {

    NB_CHECK(nNew >= 0, "append_kv(): new token number must be >= 0.");
    if (nNew == 0) return true;

    const int start = table.num_tokens(); 
    if (!table.ensure_capacity(start + nNew)) return false;

    cache.write(layer, table, start, kNew, vNew, nNew);
    table.append_tokens(nNew);
    return true;
}
}