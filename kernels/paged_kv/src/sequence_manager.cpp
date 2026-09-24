#include "sequence_manager.h"
#include "check.h"

namespace paged_kv {

bool prepare_sequence(BlockTable& table, int n) {
    NB_CHECK(n >= 0, "prepare_sequence(): n must be >= 0.");
    if (n == 0) return true;

    const int start = table.num_tokens();          // 推进前
    if (!table.ensure_capacity(start + n)) return false;
    table.append_tokens(n);                         // 提交逻辑长度
    return true;
}

}  // namespace paged_kv