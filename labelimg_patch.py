#!/usr/bin/env python3
"""
LabelImg compatibility patch for Python 3.14.
Fixes the setValue float/int type error.
"""

import sys
import importlib.util

def patch_labelimg():
    """Patch labelImg to fix Python 3.14 compatibility."""
    
    try:
        # Import labelImg
        import labelImg.labelImg as labelimg_module
        
        # Get the original scroll_request method
        original_scroll_request = labelimg_module.MainWindow.scroll_request
        
        def patched_scroll_request(self, delta, orientation):
            """Patched scroll_request that handles float values correctly."""
            try:
                if orientation == 1:  # Qt.Vertical
                    bar = self.scroll_area.verticalScrollBar()
                else:  # Qt.Horizontal  
                    bar = self.scroll_area.horizontalScrollBar()
                
                units = delta / 8  # Original calculation
                # Fix: Convert float to int for setValue
                new_value = int(bar.value() + bar.singleStep() * units)
                bar.setValue(new_value)
                
            except Exception as e:
                print(f"Scroll error (patched): {e}")
                # Fallback to small scroll increment
                try:
                    if orientation == 1:
                        bar = self.scroll_area.verticalScrollBar()
                    else:
                        bar = self.scroll_area.horizontalScrollBar()
                    
                    # Simple increment/decrement
                    if delta > 0:
                        bar.setValue(bar.value() + bar.singleStep())
                    else:
                        bar.setValue(bar.value() - bar.singleStep())
                except:
                    pass  # Ignore if this fails too
        
        # Apply the patch
        labelimg_module.MainWindow.scroll_request = patched_scroll_request
        
        print("✅ Applied labelImg compatibility patch for Python 3.14")
        return True
        
    except Exception as e:
        print(f"❌ Failed to patch labelImg: {e}")
        return False

def run_patched_labelimg(image_dir):
    """Run labelImg with the compatibility patch applied."""
    
    if not patch_labelimg():
        return False
    
    try:
        import labelImg.labelImg as labelimg_module
        import sys
        
        # Prepare arguments for labelImg
        original_argv = sys.argv.copy()
        sys.argv = ['labelImg', image_dir]
        
        print(f"🏷️  Starting patched labelImg for: {image_dir}")
        
        # Run labelImg main function
        labelimg_module.main()
        
        # Restore original argv
        sys.argv = original_argv
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to run patched labelImg: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python labelimg_patch.py <image_directory>")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    
    print("🔧 LabelImg Python 3.14 Compatibility Patch")
    print("=" * 50)
    
    success = run_patched_labelimg(image_dir)
    
    if not success:
        print("\n❌ Patched labelImg failed to start")
        print("💡 Alternative: Use custom annotation tool")
        print(f"   Command: python annotation_tool.py {image_dir}")