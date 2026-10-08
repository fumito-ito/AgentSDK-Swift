import Foundation

/// The stack size used for deeply recursive JSON processing.
let largeStackSize = 64 * 1024 * 1024

/// Runs `body` synchronously on a dedicated thread with a large stack.
///
/// Swift: upstream relies on Python's recursion limit (~1000 frames) for recursive JSON schema
/// processing. Swift concurrency threads have small stacks (512 KiB), and nested JSON values are
/// also released recursively, so deep schemas that Python accepts would overflow the stack. See
/// porting/decisions.md D9.
func runOnLargeStack<T>(_ body: () throws -> T) rethrows -> T {
    try withoutActuallyEscaping(body) { escapable in
        // The caller blocks until the thread finishes, so the body never outlives this call.
        let box = LargeStackResultBox<T>(body: escapable)
        let semaphore = DispatchSemaphore(value: 0)
        let thread = Thread {
            box.run()
            semaphore.signal()
        }
        thread.stackSize = largeStackSize
        thread.start()
        semaphore.wait()
        // swiftlint:disable:next force_unwrapping
        return try box.result!.get()
    }
}

private final class LargeStackResultBox<T>: @unchecked Sendable {
    private var body: (() throws -> T)?
    var result: Result<T, any Error>?

    init(body: @escaping () throws -> T) {
        self.body = body
    }

    func run() {
        guard let body else { return }
        result = Result { try body() }
        // Drop the body before signalling so `withoutActuallyEscaping` sees it released.
        self.body = nil
    }
}
