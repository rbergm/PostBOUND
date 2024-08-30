SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  votes AS v,
  users AS u
WHERE u.Id = p.OwnerUserId
  AND p.Id = v.PostId
  AND p.Id = c.PostId
  AND c.Score = 0
  AND c.CreationDate <= CAST('2014-09-10 02:47:53' AS timestamp)
  AND p.Score >= 0
  AND p.Score <= 19
  AND p.CommentCount <= 10
  AND p.CreationDate <= CAST('2014-08-28 13:31:33' AS timestamp)
  AND v.CreationDate <= CAST('2014-09-12 00:00:00' AS timestamp)
  AND u.DownVotes >= 0;
