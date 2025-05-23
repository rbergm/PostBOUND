SELECT COUNT(*)
FROM comments AS c, posts AS p, users AS u
WHERE c.UserId = u.Id
  AND u.Id = p.OwnerUserId
  AND c.Score = 0
  AND p.Score >= 0
  AND p.Score <= 15
  AND p.ViewCount >= 0
  AND p.ViewCount <= 3002
  AND p.AnswerCount <= 3
  AND p.CommentCount <= 10
  AND u.DownVotes <= 0
  AND u.UpVotes >= 0
  AND u.CreationDate >= CAST('2010-08-23 16:21:10' AS timestamp)
  AND u.CreationDate <= CAST('2014-09-02 09:50:06' AS timestamp);
