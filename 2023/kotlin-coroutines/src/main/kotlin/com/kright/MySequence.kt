package com.kright

import kotlin.coroutines.Continuation
import kotlin.coroutines.CoroutineContext
import kotlin.coroutines.EmptyCoroutineContext
import kotlin.coroutines.intrinsics.COROUTINE_SUSPENDED
import kotlin.coroutines.intrinsics.createCoroutineUnintercepted
import kotlin.coroutines.intrinsics.suspendCoroutineUninterceptedOrReturn
import kotlin.coroutines.resume

interface MySequenceScope<T> {
    suspend fun yield(value: T)
}

class MySequence<T>(private val fn: suspend MySequenceScope<T>.() -> Unit) : Sequence<T> {
    override fun iterator(): Iterator<T> {
        return MySequenceIterator(fn)
    }
}

class MySequenceIterator<T>(fn: suspend MySequenceScope<T>.() -> Unit) : Iterator<T> {
    private var nextIsComputed: Boolean = false
    private var next: T? = null
    private var fnContinuation: Continuation<Unit>? = null

    private val parentContinuation = object : Continuation<Unit> {
        override val context: CoroutineContext
            get() = EmptyCoroutineContext

        override fun resumeWith(result: Result<Unit>) {
            nextIsComputed = false
            next = null
            fnContinuation = null
        }
    }

    private val sequenceScope = object : MySequenceScope<T> {
        override suspend fun yield(value: T) {
            next = value
            nextIsComputed = true
            suspendCoroutineUninterceptedOrReturn { c ->
                fnContinuation = c
                COROUTINE_SUSPENDED
            }
        }
    }

    init {
        fnContinuation = fn.createCoroutineUnintercepted(sequenceScope, parentContinuation)
    }

    private fun computeNextIfNecessary() {
        if (!nextIsComputed) {
            fnContinuation?.resume(Unit)
        }
    }

    override fun hasNext(): Boolean {
        computeNextIfNecessary()
        return nextIsComputed
    }

    override fun next(): T {
        computeNextIfNecessary()
        if (nextIsComputed) {
            nextIsComputed = false
            return next!!
        } else {
            throw NoSuchElementException()
        }
    }
}