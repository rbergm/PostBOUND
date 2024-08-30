SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postHistory AS ph,
  votes AS v,
  badges AS b,
  users AS u
WHERE u.Id = p.OwnerUserId
  AND u.Id = b.UserId
  AND p.Id = c.PostId
  AND p.Id = ph.PostId
  AND p.Id = v.PostId
  AND c.Score = 0
  AND c.CreationDate <= CAST('2014-09-09 19:58:29' AS timestamp)
  AND p.Score >= -4
  AND p.ViewCount >= 0
  AND p.ViewCount <= 5977
  AND p.AnswerCount <= 4
  AND p.CommentCount >= 0
  AND p.CommentCount <= 11
  AND p.CreationDate >= CAST('2011-01-25 08:31:41' AS timestamp)
  AND u.Reputation <= 312
  AND u.DownVotes <= 0;
