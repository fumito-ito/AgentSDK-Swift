import Foundation
import Testing
@testable import Agents

// Ported from tests/test_disabled_trace_provider_shutdown.py and
// tests/test_disabled_trace_provider_flush.py (one Swift file per pair, D10 naming).
//
// Swift: `flush_traces()` reads the global provider, which suites running in parallel share, so the
// flush test calls `forceFlush()` on the provider that `flush_traces()` would use.
@Suite struct DisabledTraceProviderTests {
    private final class CountingProcessor: TracingProcessor, @unchecked Sendable {
        let shutdownCalls = SharedBox(0)
        let flushCalls = SharedBox(0)

        func onTraceStart(_ trace: any Trace) {}
        func onTraceEnd(_ trace: any Trace) {}
        func onSpanStart(_ span: any Span) {}
        func onSpanEnd(_ span: any Span) {}
        func shutdown(timeout: Double?) { shutdownCalls.mutate { $0 += 1 } }
        func forceFlush() { flushCalls.mutate { $0 += 1 } }
    }

    @Test func testDisabledTraceProviderStillShutsDownRegisteredProcessors() {
        let provider = DefaultTraceProvider()
        let processor = CountingProcessor()
        provider.registerProcessor(processor)
        provider.setDisabled(true)

        provider.shutdown(timeout: nil)

        #expect(processor.shutdownCalls.value == 1)
    }

    @Test func testFlushTracesStillFlushesRegisteredProcessorsWhenDisabled() {
        let provider = DefaultTraceProvider()
        let processor = CountingProcessor()
        provider.registerProcessor(processor)
        provider.setDisabled(true)

        provider.forceFlush()

        #expect(processor.flushCalls.value == 1)
    }
}
