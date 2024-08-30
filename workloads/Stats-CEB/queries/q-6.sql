SELECT COUNT(*)
FROM badges AS b, posts AS p
WHERE b.UserId = p.OwnerUserId
  AND b.Date <= CAST('2014-09-11 08:55:52' AS timestamp)
  AND p.AnswerCount >= 0
  AND p.AnswerCount <= 4
  AND p.CommentCount >= 0
  AND p.CommentCount <= 17;
