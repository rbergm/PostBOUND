SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postLinks AS pl,
  votes AS v,
  badges AS b,
  users AS u
WHERE u.Id = p.OwnerUserId
  AND p.Id = pl.RelatedPostId
  AND p.Id = v.PostId
  AND p.Id = c.PostId
  AND u.Id = b.UserId
  AND p.Score >= -1
  AND p.Score <= 14
  AND pl.CreationDate <= CAST('2014-06-25 13:05:06' AS timestamp)
  AND v.CreationDate >= CAST('2009-02-02 00:00:00' AS timestamp)
  AND b.Date >= CAST('2010-08-04 08:50:31' AS timestamp)
  AND b.Date <= CAST('2014-09-02 02:51:22' AS timestamp)
  AND u.DownVotes >= 0;
