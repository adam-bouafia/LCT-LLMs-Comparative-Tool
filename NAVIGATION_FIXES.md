# Navigation Fixes - API Keys Manager

## Issue Report
User reported that when selecting option 4 ("Show API Key") in the API Keys Manager, there was no way to cancel or go back when prompted for the service selection.

**User Experience:**
```
Choose an option [0/1/2/3/4/5/6]: 4
Which service? [openai/huggingface/anthropic]: huggingface  ← Typo
Please select one of the available options
Which service? [openai/huggingface/anthropic]: 2  ← Wrong
Please select one of the available options  
Which service? [openai/huggingface/anthropic]: 0  ← Trying to exit
Please select one of the available options  ← STUCK!
```

## Fixes Applied

### Option 4: Show API Key (Line 2641)
**Before:**
```python
service = Prompt.ask(
    "Which service?", choices=["openai", "huggingface", "anthropic"]
)
```

**After:**
```python
service = Prompt.ask(
    "Which service? (0 to cancel)", 
    choices=["openai", "huggingface", "anthropic", "0"]
)

if service == "0":
    continue
```

### Option 5: Remove API Key (Line 2673)
**Before:**
```python
service = Prompt.ask("Which service to remove?", choices=services)
```

**After:**
```python
# Add "0" to choices for back option
choices_with_back = list(services) + ["0"]
service = Prompt.ask(
    "Which service to remove? (0 to cancel)", 
    choices=choices_with_back
)

if service == "0":
    continue
```

## Options Review

### Options 1-3: Add/Update API Keys
✅ **Already handled correctly** - These options use `Prompt.ask()` with empty default values. Users can simply press Enter to cancel, which triggers:
```python
if api_key:
    # Save the key
else:
    self.console.print("[yellow]No key entered, operation cancelled[/yellow]")
```

### Option 6: List All Configured Keys
✅ **No action needed** - Display-only option with no prompts, just shows information and returns to menu.

## Testing Checklist

- [x] Option 1: Add OpenAI API Key - Press Enter cancels (existing behavior)
- [x] Option 2: Add HuggingFace Token - Press Enter cancels (existing behavior)
- [x] Option 3: Add Anthropic API Key - Press Enter cancels (existing behavior)
- [ ] Option 4: Show API Key - Type "0" to cancel (NEW FIX)
- [ ] Option 5: Remove API Key - Type "0" to cancel (NEW FIX)
- [x] Option 6: List Keys - No prompts, just displays (no change needed)

## Navigation Consistency

All menu options now have consistent navigation patterns:
- **Main menu**: Option "0" returns to main menu ✅
- **Option 4 & 5**: "0" cancels service selection ✅
- **Options 1-3**: Empty Enter cancels key input ✅
- **Option 6**: Display only, returns automatically ✅

## User Experience Improvement

**Before:** Users could get stuck in service selection prompts with no escape
**After:** Users can always exit with "0" or empty Enter, providing consistent UX

## File Modified
- `app/src/ui/interactive_lct.py` - API Keys Manager section (lines 2425-2710)
