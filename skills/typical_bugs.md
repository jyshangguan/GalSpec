# Typical Bugs in GalSpec Spectral Fitting

This file documents common bugs, mistakes, and pitfalls in GalSpec spectral fitting. Each bug includes:
- **The Bug**: What went wrong
- **Detection**: How to spot it
- **The Fix**: Correct code/approach
- **Why It Happens**: Root cause

---

## Bug #1: Tying a Parameter to Itself

### The Bug
When tying multiple parameters together, accidentally including the reference parameter in the loop causes it to be tied to itself, effectively fixing its value.

### Example of WRONG Code:
```python
# ❌ WRONG - includes 'nHalpha' in the loop!
for ln in ['nHalpha', 'NII_6583', 'NII_6548']:
    model[ln].dv.tied = galspec.tie_template_dv('nHalpha')
```

**What Happens**:
- `nHalpha.dv.tied = tie_template_dv('nHalpha')`
- This ties nHalpha to **itself**, making it fixed at initial value
- Result: dv stays at 0, cannot vary during fit

### Example of CORRECT Code:
```python
# ✅ CORRECT - skip the reference parameter!
for ln in ['NII_6583', 'NII_6548']:  # ← Don't include 'nHalpha'!
    model[ln].dv.tied = galspec.tie_template_dv('nHalpha')
```

**What Happens**:
- `nHalpha.dv` is **free to vary** (the master parameter)
- `NII_6583.dv` tied to nHalpha
- `NII_6548.dv` tied to nHalpha
- Result: All three share ONE free parameter (nHalpha.dv)

### Detection:
- **Symptom**: Parameter stays at initial value (e.g., dv=0) despite fitting
- **Symptom**: χ² much higher than expected
- **Check**: Print parameter values before and after fit
- **Check**: Verify `model['param_name'].tied` is not the same as param_name

### Impact:
- **Before fix**: χ²/ν = 12.9, narrow dv fixed at 0
- **After fix**: χ²/ν = 1.6, narrow dv free to vary (58 km/s)

### Common Scenarios:
This bug often occurs when:
1. Tying narrow line velocities together
2. Tying doublet amplitude ratios
3. Tying kinematic parameters in multi-component models

### General Rule:
**NEVER include the reference parameter in the tying loop**
```python
# Pattern: Reference is FIRST, others tied to it
reference_param = 'nHalpha'
for param in ['other1', 'other2', 'other3']:  # ← No reference!
    model[param].some_tied = tie_function(reference_param)
```

### Why It Happens:
- It's an easy mistake when iterating through a list of parameter names
- The tied parameter returns itself when tied to itself
- This creates a circular reference that fixes the value
- Astropy's fitting doesn't warn about this explicitly

---

## Template for Future Bugs

Use this template to add new bugs:

```markdown
## Bug #[N]: [Short Title]

### The Bug
[Brief description of what went wrong]

### Example of WRONG Code:
```python
# ❌ WRONG
[bad code example]
```

**What Happens**:
[Explanation of the problem]

### Example of CORRECT Code:
```python
# ✅ CORRECT
[good code example]
```

**What Happens**:
[Explanation of why it works]

### Detection:
- **Symptom**: [How to recognize the bug]
- **Check**: [How to verify]
- **Test**: [Code to test for the bug]

### Impact:
[Before and after comparison if possible]

### Common Scenarios:
[When this bug typically occurs]

### Why It Happens:
[Root cause explanation]

### Prevention:
[How to avoid this bug in the future]
```

---

## Maintenance Notes

**When to add a new bug**:
- You encounter a bug that took >10 minutes to debug
- The bug is subtle and could be easily repeated
- The bug has a clear lesson for future users

**What makes a good bug entry**:
- Clear before/after comparison
- Code examples that can be copied/pasted
- Detection checklist
- Prevention tips

**Version history**:
- 2025-03-20: Initial file with Bug #1 (dv tying bug)
