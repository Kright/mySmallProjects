package com.github.kright


import scala.concurrent.{ExecutionContext, Future}
import scala.util.Random

//class CleverReplier(replier: Replier)(implicit config: ConfigAdminId with ConfigBotId, perf: PerformanceMeasurerFactory) extends MessageHandler {
//  private val replierPerf = perf("CleverReplier")
//  private val ignoredUsers = Set(133330829L)
//
//  override def apply(message: Message)(implicit request: RequestHandler[Future], ec: ExecutionContext): Result = {
//    if (!needToReply(message)) {
//      return noReply
//    }
//
//    message.text.map{ text =>
//      val (reply, log) = replierPerf {
//        replier(text, perf)
//      }
//      reply match {
//        case Some(replyText) => {
//          doReply(
//            SendMessage(
//              chatId = config.adminId,
//              text = s"user: ${message.from}\n${log}\n${replierPerf.lastTimeNs}ns",
//              disableNotification = Option(true),
//              replyToMessageId = None
//            ),
//            SendMessage(
//              chatId = message.chat.id,
//              text = replyText,
//              disableNotification = Option(true),
//              replyToMessageId = Option(message.messageId)
//            )
//          )
//        }
//        case None => {
//          noReply
//        }
//      }
//    }.getOrElse(noReply)
//  }
//
//  def needToReply(message: Message): Boolean = {
//    if (message.chat.isPrivate) return true
//
//    if (message.chat.isGroup) {
//      if (message.chat.id == Groups.electroKorolev) {
//        if (message.from.exists(user => ignoredUsers.contains(user.id))) return false
//
//        if (message.replyToMessage.flatMap(_.from).exists(_.id == config.botId)) return true
//        if (message.text.exists(_.toLowerCase.contains("бот")) && Random.nextInt(2) == 0) return true
//        return Random.nextInt(10) == 0
//      }
//
//      return false
//    }
//
//    false
//  }
//}
