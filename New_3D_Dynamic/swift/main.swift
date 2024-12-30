import Cocoa

class ClickThroughImageView: NSImageView {
    override func hitTest(_ point: NSPoint) -> NSView? {
        // Ignore mouse events for the image view
        return nil
    }
}

@NSApplicationMain
class AppDelegate: NSObject, NSApplicationDelegate {

    var window: NSWindow!
    var imageIndex = 0
    let imageNames = ["bread.jpeg", "bread.jpeg", "angleImage(1).jpg"] // Add your image names here

    func applicationDidFinishLaunching(_ aNotification: Notification) {
        
        
        
        let screenSize = NSScreen.main?.frame.size ?? CGSize(width: 800, height: 600)
        let windowRect = NSRect(x: 0, y: 0, width: 200, height: 200)
        
        window = NSWindow(contentRect: windowRect, styleMask: [.borderless], backing: .buffered, defer: false)
        window.isOpaque = false
        window.backgroundColor = NSColor.clear
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.level = .floating
        window.ignoresMouseEvents = true // Make the entire window click-through

        
//        var imageView = NSImageView(frame: windowRect)
        var imageView = ClickThroughImageView(frame: windowRect)

        imageView.imageScaling = .scaleAxesIndependently

//        imageView = NSImageView(frame: window.contentView!.bounds)
        imageView = ClickThroughImageView(frame: windowRect)
        imageView.image = NSImage(named: "bread2.png") // Replace with the actual image name
        imageView.alphaValue = 1.5
        
        NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { (event) in
            self.handleGlobalKeyPress(event, imageView:imageView)
                }

                // Add the image view to the window's content view
                window.contentView = imageView

//                window.contentView?.addSubview(imageView)


        window.makeKeyAndOrderFront(nil)
        
        // Add tracking area to follow the cursor
//        let trackingArea = NSTrackingArea(rect: windowRect, options: [.mouseMoved, .activeAlways], owner: self, userInfo: nil)
//        window.contentView?.addTrackingArea(trackingArea)
        
//         Start a timer to change the image every 2 seconds (adjust as needed)
        Timer.scheduledTimer(timeInterval: 0.01, target: self, selector: #selector(moveImage), userInfo: nil, repeats: true)
    }

    @objc func moveImage(with event: NSEvent) {
        let currentMouseLocation = NSEvent.mouseLocation
        window.setFrameOrigin(NSPoint(x: currentMouseLocation.x - 100, y: currentMouseLocation.y - 100))
        
    
    }
    func handleGlobalKeyPress(_ event: NSEvent, imageView: NSImageView) {
        print("before big if")
        if event.modifierFlags.contains(.command) && event.modifierFlags.contains(.shift) && event.charactersIgnoringModifiers == "V" {
            // Command+Shift+V is pressed
            print("before if")
            if imageView.alphaValue < 0.001{
                print("inside if")
                imageView.alphaValue = 0.1
            }else{
                imageView.alphaValue = 0
            }
            // Handle your logic here

            // Returning nil stops the event from being processed further
            
        }
    }
    
    // Handle mouse movement to update the window position
//     func mouseMoved(with event: NSEvent) {
//        let cursorPoint = NSEvent.mouseLocation
//        window.setFrameOrigin(NSPoint(x: cursorPoint.x - window.frame.width / 2, y: cursorPoint.y - window.frame.height / 2))
//    }
    
    // Change the image to the next one in the list
//    @objc func changeImage() {
//        let imageView = NSImageView(frame: window.contentView!.bounds)
//                imageView.image = NSImage(named: "bread.jpeg") // Replace with the actual image name
//
//                // Add the image view to the window's content view
//                window.contentView?.addSubview(imageView)
//    }
}
