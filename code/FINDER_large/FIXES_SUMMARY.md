# Bug Fixes Summary

## Problem
The program was crashing when calling `nStepReplayMem.Add()` during the training process.

## Root Cause
The main issue was in `code/FINDER_large/src/lib/nstep_replay_mem.cpp` in the `Add` method:

1. **Vector Access Without Allocation**: The `gamma_pow` vector was declared but not properly sized before accessing elements, causing memory access violations.

2. **Out-of-Bounds Access**: The code was accessing `gamma_pow[T-i]` where `T-i` could be larger than the allocated size.

3. **Missing Field Copy**: The `rollout_return` field was not being copied from Python to C++ in the Cython wrapper.

## Fixes Applied

### 1. Fixed Vector Allocation (nstep_replay_mem.cpp)
**Before:**
```cpp
std::vector <double> gamma_pow;
double gamma_n = 1;
for(int i = 0;i<=n_step;i++){
    gamma_pow[i] = gamma_n;  // BUG: Accessing unallocated memory
    gamma_n *= gamma;
}
```

**After:**
```cpp
int max_power = std::max(n_step, T);
std::vector <double> gamma_pow(max_power + 1);  // Properly allocate memory
double gamma_n = 1;
for(int i = 0; i <= max_power; i++){
    gamma_pow[i] = gamma_n;
    gamma_n *= gamma;
}
```

### 2. Added Missing Field Copy (nstep_replay_mem.pyx)
**Added:**
```python
deref(self.inner_MvcEnv).rollout_return = mvcenv.rollout_return
```

### 3. Added Safety Checks and Debug Output
**Added:**
```cpp
// Debug output
printf("DEBUG: T = %d, n_step = %d\n", T, n_step);
printf("DEBUG: reward_seq.size() = %zu\n", env->reward_seq.size());
printf("DEBUG: act_seq.size() = %zu\n", env->act_seq.size());
printf("DEBUG: sum_rewards.size() = %zu\n", env->sum_rewards.size());

// Safety checks
assert(env->reward_seq.size() == T);
assert(env->act_seq.size() == T);
assert(env->sum_rewards.size() >= T);
```

### 4. Fixed Missing Method Declaration (mvc_env.h)
**Added:**
```cpp
void printGraph();  // Uncommented the declaration
```

### 5. Added Method to Cython Wrapper (mvc_env.pyx)
**Added:**
```python
def printGraph(self):
    deref(self.inner_MvcEnv).printGraph()
```

## How to Apply Fixes

1. Run the rebuild script:
   ```bash
   python rebuild_and_test.py
   ```

2. Or manually rebuild:
   ```bash
   python setup.py build_ext --inplace
   ```

## Expected Behavior After Fixes
- The program should no longer crash when calling `nStepReplayMem.Add()`
- Debug output will help identify any remaining issues
- The training process should continue normally