SELECT COUNT(*)
FROM comments AS c, posts AS p, users AS u
WHERE u.Id = p.OwnerUserId
  AND c.UserId = u.Id
  AND c.CreationDate >= CAST('2010-07-27 17:46:38' AS timestamp)
  AND p.AnswerCount >= 0
  AND p.AnswerCount <= 4
  AND p.CommentCount >= 0
  AND p.CommentCount <= 11
  AND p.CreationDate >= CAST('2010-07-26 09:46:48' AS timestamp)
  AND p.CreationDate <= CAST('2014-09-13 10:09:50' AS timestamp)
  AND u.Reputation >= 1
  AND u.CreationDate >= CAST('2010-08-03 19:42:40' AS timestamp)
  AND u.CreationDate <= CAST('2014-09-12 02:20:03' AS timestamp);
