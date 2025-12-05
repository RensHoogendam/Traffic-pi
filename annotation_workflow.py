#!/usr/bin/env python3
"""
Annotation workflow manager for traffic light datasets.
Provides multiple annotation options and handles compatibility issues.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_labelimg_compatibility():
    """Check if labelImg works with current Python version."""
    
    try:
        import labelImg
        print("✅ labelImg imported successfully")
        return True
    except Exception as e:
        print(f"❌ labelImg import failed: {e}")
        return False

def run_custom_annotator(images_dir, labels_dir=None):
    """Run the custom annotation tool."""
    
    if labels_dir is None:
        labels_dir = f"{images_dir}_labels"
    
    print(f"🚀 Starting custom annotation tool...")
    print(f"📁 Images: {images_dir}")
    print(f"📁 Labels: {labels_dir}")
    
    # Run the custom annotator
    try:
        from annotation_tool import TrafficLightAnnotator
        annotator = TrafficLightAnnotator(images_dir, labels_dir)
        annotator.run()
        return True
    except Exception as e:
        print(f"❌ Custom annotator failed: {e}")
        return False

def run_labelimg_safe(images_dir):
    """Try to run labelImg with error handling."""
    
    print(f"🏷️  Attempting to run labelImg...")
    
    try:
        # Try different approaches to run labelImg
        commands_to_try = [
            ["labelImg", images_dir],
            ["python", "-m", "labelImg.labelImg", images_dir],
            ["python", "-c", f"import labelImg.labelImg; labelImg.labelImg.main('{images_dir}')"]
        ]
        
        for cmd in commands_to_try:
            try:
                print(f"Trying command: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=False, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ labelImg started successfully")
                    return True
                else:
                    print(f"Command failed with return code {result.returncode}")
                    if result.stderr:
                        print(f"Error: {result.stderr}")
            
            except Exception as e:
                print(f"Command failed: {e}")
                continue
        
        return False
        
    except Exception as e:
        print(f"❌ Failed to run labelImg: {e}")
        return False

def setup_coco_annotator():
    """Set up web-based COCO annotator as alternative."""
    
    print("🌐 Setting up web-based annotation alternative...")
    
    try:
        # Check if we can use a web-based annotator
        import webbrowser
        
        # Create a simple HTML annotation interface
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Traffic Light Annotator</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .info { background: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .option { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .command { background: #263238; color: #ffffff; padding: 10px; border-radius: 3px; font-family: monospace; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚦 Traffic Light Annotation Options</h1>
        
        <div class="info">
            <h3>labelImg Compatibility Issue Detected</h3>
            <p>Your Python 3.14 version has compatibility issues with labelImg. Here are your options:</p>
        </div>
        
        <div class="option">
            <h3>🎯 Option 1: Custom Annotation Tool (Recommended)</h3>
            <p>Use our custom-built annotation tool with Python 3.14 compatibility:</p>
            <div class="command">python annotation_tool.py training_data/extracted_frames</div>
            <p><strong>Features:</strong> Simple interface, YOLO format output, keyboard shortcuts</p>
        </div>
        
        <div class="option">
            <h3>🏷️ Option 2: Fixed labelImg</h3>
            <p>Try the patched labelImg version:</p>
            <div class="command">python annotation_workflow.py --labelimg training_data/extracted_frames</div>
            <p><strong>Features:</strong> Full labelImg functionality with compatibility fixes</p>
        </div>
        
        <div class="option">
            <h3>🌐 Option 3: Web-based Annotation</h3>
            <p>Use online annotation tools:</p>
            <ul>
                <li><a href="https://www.makesense.ai/" target="_blank">makesense.ai</a> - Free online tool</li>
                <li><a href="https://roboflow.com/" target="_blank">Roboflow</a> - Advanced features</li>
                <li><a href="https://labelbox.com/" target="_blank">Labelbox</a> - Professional solution</li>
            </ul>
        </div>
        
        <div class="option">
            <h3>📋 Controls for Custom Tool</h3>
            <ul>
                <li><strong>Left-click + drag:</strong> Draw bounding box</li>
                <li><strong>1-4 keys:</strong> Switch traffic light class (red/yellow/green/unknown)</li>
                <li><strong>Space/Enter:</strong> Save and next image</li>
                <li><strong>Backspace:</strong> Previous image</li>
                <li><strong>U:</strong> Undo last annotation</li>
                <li><strong>C:</strong> Clear all annotations</li>
                <li><strong>S:</strong> Save current annotations</li>
                <li><strong>Q:</strong> Quit</li>
            </ul>
        </div>
    </div>
</body>
</html>
        """
        
        # Save HTML file
        html_path = Path("annotation_options.html")
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"📄 Created annotation guide: {html_path}")
        print(f"🌐 Opening in browser...")
        
        # Open in browser
        webbrowser.open(f"file://{html_path.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Could not set up web interface: {e}")
        return False

def main():
    """Main workflow manager."""
    
    print("🚦 Traffic Light Annotation Workflow Manager")
    print("=" * 60)
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python annotation_workflow.py [--labelimg|--custom|--web] <images_directory>")
        print("Example: python annotation_workflow.py training_data/extracted_frames")
        print("         python annotation_workflow.py --custom training_data/extracted_frames")
        return
    
    # Determine mode and directory
    if sys.argv[1].startswith('--'):
        mode = sys.argv[1][2:]  # Remove --
        images_dir = sys.argv[2] if len(sys.argv) > 2 else "training_data/extracted_frames"
    else:
        mode = "auto"
        images_dir = sys.argv[1]
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    # Count images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_count = 0
    for ext in image_extensions:
        image_count += len(list(Path(images_dir).glob(f'*{ext}')))
        image_count += len(list(Path(images_dir).glob(f'*{ext.upper()}')))
    
    print(f"📁 Images directory: {images_dir}")
    print(f"📊 Found {image_count} images")
    
    if image_count == 0:
        print("❌ No images found! Make sure you have extracted frames first.")
        print("💡 Try: python collect_data.py --extract <video_file>")
        return
    
    # Execute based on mode
    success = False
    
    if mode == "custom":
        print("\n🎯 Running custom annotation tool...")
        success = run_custom_annotator(images_dir)
    
    elif mode == "labelimg":
        print("\n🏷️  Attempting labelImg...")
        if check_labelimg_compatibility():
            success = run_labelimg_safe(images_dir)
        else:
            print("⚠️  labelImg compatibility issues detected")
            print("🔄 Falling back to custom tool...")
            success = run_custom_annotator(images_dir)
    
    elif mode == "web":
        print("\n🌐 Setting up web-based annotation...")
        success = setup_coco_annotator()
    
    else:  # auto mode
        print("\n🔍 Auto-detecting best annotation method...")
        
        # Try labelImg first (if compatible)
        if check_labelimg_compatibility():
            print("✅ labelImg is compatible, trying to launch...")
            success = run_labelimg_safe(images_dir)
            
            if not success:
                print("⚠️  labelImg failed to start, using custom tool...")
                success = run_custom_annotator(images_dir)
        else:
            print("⚠️  labelImg incompatible with Python 3.14")
            print("🔄 Using custom annotation tool...")
            success = run_custom_annotator(images_dir)
    
    # Final status
    if success:
        print(f"\n✅ Annotation workflow completed successfully!")
        print(f"📁 Check labels in: {images_dir}_labels/")
        print(f"🚀 Next step: python train_model.py")
    else:
        print(f"\n❌ Annotation workflow failed.")
        print(f"💡 Try manual options:")
        print(f"   Custom tool: python annotation_tool.py {images_dir}")
        print(f"   Web option:  python annotation_workflow.py --web")

if __name__ == "__main__":
    main()